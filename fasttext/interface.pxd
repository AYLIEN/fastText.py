# fastText C++ interface
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr

cdef extern from "interface.h":
    cdef cppclass FastTextModel:
        FastTextModel()
        vector[vector[string]] classifierPredictProb(string text, int32_t k)

        # Wrapper for Dictionary class
        int32_t dictGetNWords()
        string dictGetWord(int32_t i)
        int32_t dictGetNLabels()
        string dictGetLabel(int32_t i)

    # Add 'except +' to the function declaration to let Cython safely raise an
    # appropriate Python exception instead
    void loadModelWrapper(string filename, FastTextModel& model) except +


