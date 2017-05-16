#define GOGUI_X(pos) (gogui_x[CORRECT_X(pos)])
#define GOGUI_Y(pos) (pure_board_size + 1 - CORRECT_Y(pos))

cdef char gogui_x[26]

cdef int string_to_integer(char *cpos )

cdef void integer_to_string(int pos, char *cpos)
