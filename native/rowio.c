#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint64_t row;
    Py_ssize_t output_index;
} row_request;

static int compare_rows(const void *left, const void *right) {
    const row_request *a = (const row_request *)left;
    const row_request *b = (const row_request *)right;
    if (a->row < b->row) {
        return -1;
    }
    if (a->row > b->row) {
        return 1;
    }
    if (a->output_index < b->output_index) {
        return -1;
    }
    if (a->output_index > b->output_index) {
        return 1;
    }
    return 0;
}

static int checked_offset(uint64_t base, uint64_t row, uint64_t row_bytes,
                          uint64_t *result) {
    if (row != 0 && row_bytes > UINT64_MAX / row) {
        return 0;
    }
    uint64_t relative = row * row_bytes;
    if (base > UINT64_MAX - relative) {
        return 0;
    }
    *result = base + relative;
    return *result <= (uint64_t)LLONG_MAX;
}

static int read_fully(int fd, char *buffer, size_t length, uint64_t offset,
                      unsigned long long *call_count, int *saved_errno) {
    if (length != 0 && offset > (uint64_t)LLONG_MAX - (length - 1)) {
        *saved_errno = EOVERFLOW;
        return 0;
    }
    size_t complete = 0;
    while (complete < length) {
        ssize_t count = pread(fd, buffer + complete, length - complete,
                              (off_t)(offset + complete));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            *saved_errno = errno;
            return 0;
        }
        ++*call_count;
        if (count == 0) {
            *saved_errno = EIO;
            return 0;
        }
        complete += (size_t)count;
    }
    return 1;
}

static PyObject *read_rows(PyObject *module, PyObject *args) {
    int fd;
    unsigned long long base_offset;
    Py_ssize_t row_bytes;
    PyObject *rows_object;
    unsigned long long total_rows;

    if (!PyArg_ParseTuple(args, "iKnOK", &fd, &base_offset, &row_bytes,
                          &rows_object, &total_rows)) {
        return NULL;
    }
    if (row_bytes < 0) {
        PyErr_SetString(PyExc_ValueError, "row_bytes must be nonnegative");
        return NULL;
    }
    if (total_rows > (unsigned long long)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "row count is too large");
        return NULL;
    }

    PyObject *rows = PySequence_Fast(rows_object, "rows must be iterable");
    if (rows == NULL) {
        return NULL;
    }
    Py_ssize_t count = PySequence_Fast_GET_SIZE(rows);
    if (count == 0) {
        Py_DECREF(rows);
        return Py_BuildValue("(yKK)", "", (unsigned long long)0,
                             (unsigned long long)0);
    }

    if (row_bytes > PY_SSIZE_T_MAX / count) {
        Py_DECREF(rows);
        PyErr_SetString(PyExc_OverflowError, "requested row data is too large");
        return NULL;
    }
    Py_ssize_t output_size = count * row_bytes;
    row_request *requests = PyMem_Malloc((size_t)count * sizeof(*requests));
    if (requests == NULL) {
        Py_DECREF(rows);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(rows, i);
        int overflow = 0;
        long long value = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow || (value < 0 && PyErr_Occurred())) {
            PyMem_Free(requests);
            Py_DECREF(rows);
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_OverflowError, "row index is too large");
            }
            return NULL;
        }
        if (value < 0 || (unsigned long long)value >= total_rows) {
            PyMem_Free(requests);
            Py_DECREF(rows);
            PyErr_SetString(PyExc_IndexError, "row outside tensor shape");
            return NULL;
        }
        requests[i].row = (uint64_t)value;
        requests[i].output_index = i;
    }
    Py_DECREF(rows);
    qsort(requests, (size_t)count, sizeof(*requests), compare_rows);

    PyObject *output = PyBytes_FromStringAndSize(NULL, output_size);
    if (output == NULL) {
        PyMem_Free(requests);
        return NULL;
    }
    char *output_buffer = PyBytes_AS_STRING(output);
    uint64_t bytes_read = 0;
    unsigned long long pread_calls = 0;
    int read_errno = 0;
    int failed = 0;

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t start = 0;
    while (start < count) {
        Py_ssize_t end = start + 1;
        while (end < count &&
               (requests[end].row == requests[end - 1].row ||
                requests[end].row == requests[end - 1].row + 1)) {
            ++end;
        }

        Py_ssize_t unique_rows = 1;
        for (Py_ssize_t i = start + 1; i < end; ++i) {
            if (requests[i].row != requests[i - 1].row) {
                ++unique_rows;
            }
        }
        if (row_bytes != 0 &&
            (size_t)unique_rows > SIZE_MAX / (size_t)row_bytes) {
            read_errno = EOVERFLOW;
            failed = 1;
            break;
        }
        size_t block_size = (size_t)unique_rows * (size_t)row_bytes;
        char *block = malloc(block_size == 0 ? 1 : block_size);
        if (block == NULL) {
            read_errno = ENOMEM;
            failed = 1;
            break;
        }
        uint64_t offset = 0;
        if (!checked_offset(base_offset, requests[start].row,
                            (uint64_t)row_bytes, &offset)) {
            free(block);
            read_errno = EOVERFLOW;
            failed = 1;
            break;
        }
        if (row_bytes != 0 &&
            !read_fully(fd, block, block_size, offset, &pread_calls,
                        &read_errno)) {
            free(block);
            failed = 1;
            break;
        }
        if (row_bytes == 0) {
            ++pread_calls;
        }
        bytes_read += block_size;
        for (Py_ssize_t i = start; i < end; ++i) {
            uint64_t row_delta = requests[i].row - requests[start].row;
            memcpy(output_buffer + requests[i].output_index * row_bytes,
                   block + row_delta * (uint64_t)row_bytes, (size_t)row_bytes);
        }
        free(block);
        start = end;
    }
    Py_END_ALLOW_THREADS

    PyMem_Free(requests);
    if (failed) {
        Py_DECREF(output);
        errno = read_errno;
        if (read_errno == ENOMEM) {
            return PyErr_NoMemory();
        }
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    return Py_BuildValue("(NKK)", output, pread_calls, bytes_read);
}

static PyMethodDef rowio_methods[] = {
    {"read_rows", read_rows, METH_VARARGS,
     "Read rows from a file descriptor and restore request order."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef rowio_module = {
    PyModuleDef_HEAD_INIT,
    "_rowio",
    "GIL-free contiguous safetensor row reads.",
    -1,
    rowio_methods,
};

PyMODINIT_FUNC PyInit__rowio(void) {
    return PyModule_Create(&rowio_module);
}
