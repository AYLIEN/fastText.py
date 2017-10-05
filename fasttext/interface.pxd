# fastText C++ interface
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr

cdef extern from "interface.h":
    cdef cppclass FastTextModel:
        FastTextModel()
        vector[vector[string]] classifierPredictProb(string text, int32_t k)

    void trainWrapper(int argc, char **argvm, int silent, void* f)

    # Add 'except +' to the function declaration to let Cython safely raise an
    # appropriate Python exception instead
    void loadModelWrapper(string filename, FastTextModel& model) except +


