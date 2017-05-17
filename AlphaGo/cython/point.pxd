cdef extern from "ray.h":
    int GOGUI_X(int pos)
    int GOGUI_Y(int pos, int pure_board_size)

    char *gogui_x

cdef int string_to_integer(char *cpos )

cdef void integer_to_string(int pos, char *cpos)
