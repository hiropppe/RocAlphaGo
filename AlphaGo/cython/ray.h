/*************
 * Pattern.h *
 *************/

#define REV18(p) (((p) >> 36) | (((p) & 0x3) << 36))
#define REV16(p) (((p) >> 32) | (((p) & 0x3) << 32))
#define REV14(p) (((p) >> 28) | (((p) & 0x3) << 28))
#define REV12(p) (((p) >> 24) | (((p) & 0x3) << 24))
#define REV10(p) (((p) >> 20) | (((p) & 0x3) << 20))
#define REV8(p) (((p) >> 16) | (((p) & 0x3) << 16))
#define REV6(p) (((p) >> 12) | (((p) & 0x3) << 12))
#define REV4(p) (((p) >> 8) | (((p) & 0x3) << 8))
#define REV2(p) (((p) >> 4) | (((p) & 0x3) << 4))

#define REV3(p) (((p) >> 4) | ((p) & 0xC) | (((p) & 0x3) << 4))
#define REV(p) (((p) >> 2) | (((p) & 0x3) << 2))

/*
#define N(board_size)   (-board_size)
#define S(board_size)   (board_size)
#define E   (1)
#define W   (-1)
#define NN(board_size)  (N+N)
#define NE(board_size)  (N+E)
#define NW(board_size)  (N+W)
#define SS(board_size)  (S+S)
#define SE(board_size)  (S+E)
#define SW(board_size)  (S+W)
#define WW  (W+W)
#define EE  (E+E)
*/

const int MD2_MAX = 16777216;	// 2^24
const int PAT3_MAX = 65536;	// 2^16

const int MD2_LIMIT = 1060624;
const int PAT3_LIMIT = 4468;

enum MD {
  MD_2,
  MD_3,
  MD_4,
  MD_MAX
};

enum LARGE_MD {
  MD_5,
  MD_LARGE_MAX
};

typedef struct pattern {
  unsigned int list[MD_MAX];
  unsigned long long large_list[MD_LARGE_MAX];
} pattern_t;


/***************
 * UctRating.h *
 ***************/
const int MOVE_DISTANCE_MAX = 15;


/*************
 * GoBoard.h *
 *************/
const int PURE_BOARD_SIZE = 19;  // 盤の大きさ

const int OB_SIZE = 2; // 盤外の幅
const int BOARD_SIZE = (PURE_BOARD_SIZE + OB_SIZE + OB_SIZE); // 盤外を含めた盤の幅

const int PURE_BOARD_MAX = (PURE_BOARD_SIZE * PURE_BOARD_SIZE); // 盤の大きさ 
const int BOARD_MAX = (BOARD_SIZE * BOARD_SIZE);                // 盤外を含めた盤の大きさ

const int MAX_STRING = (PURE_BOARD_MAX * 4 / 5); // 連の最大数 
const int MAX_NEIGHBOR = MAX_STRING;             // 隣接する敵連の最大数

const int BOARD_START = OB_SIZE;                        // 盤の始点  
const int BOARD_END = (PURE_BOARD_SIZE + OB_SIZE - 1);  // 盤の終点  

const int STRING_LIB_MAX = (BOARD_SIZE * (PURE_BOARD_SIZE + OB_SIZE));  // 1つの連の持つ呼吸点の最大数
const int STRING_POS_MAX = (BOARD_SIZE * (PURE_BOARD_SIZE + OB_SIZE));  // 連が持ちうる座標の最大値

const int STRING_END = (STRING_POS_MAX - 1); // 連の終端を表す値
const int NEIGHBOR_END = (MAX_NEIGHBOR - 1);  // 隣接する敵連の終端を表す値
const int LIBERTY_END = (STRING_LIB_MAX - 1); // 呼吸点の終端を表す値

const int MAX_RECORDS = (PURE_BOARD_MAX * 3); // 記録する着手の最大数 
const int MAX_MOVES = (MAX_RECORDS - 1);      // 着手数の最大値

const int PASS = 0;     // パスに相当する値
const int RESIGN = -1;  // 投了に相当する値

const double KOMI = 6.5; // デフォルトのコミの値

const int MAX_POLICY_FEATURE = 48;

#define POS(x, y, board_size) ((x) + (y) * (board_size))
#define X(pos, board_size) ((pos) % board_size)
#define Y(pos, board_size) ((pos) / board_size)

#define CORRECT_X(pos, board_size, ob_size) ((pos) % board_size - ob_size)
#define CORRECT_Y(pos, board_size, ob_size) ((pos) / board_size - ob_size)

#define NORTH(pos, board_size) ((pos) - board_size)
#define WEST(pos) ((pos) - 1)
#define EAST(pos) ((pos) + 1)
#define SOUTH(pos, board_size) ((pos) + board_size)

#define NORTH_WEST(pos, board_size) ((pos) - board_size - 1)
#define NORTH_EAST(pos, board_size) ((pos) - board_size + 1)
#define SOUTH_WEST(pos, board_size) ((pos) + board_size - 1)
#define SOUTH_EAST(pos, board_size) ((pos) + board_size + 1)

#define FLIP_COLOR(col) ((col) ^ 0x3)

#define DX(pos1, pos2, board_x)  (abs(board_x[(pos1)] - board_x[(pos2)]))
#define DY(pos1, pos2, board_y)  (abs(board_y[(pos1)] - board_y[(pos2)]))
#define DIS(pos1, pos2, move_dis) (move_dis[DX(pos1, pos2)][DY(pos1, pos2)])


enum stone {
    S_EMPTY,
    S_BLACK,
    S_WHITE,
    S_OB,
    S_MAX
};

enum eye_condition {
  E_NOT_EYE,           // 眼でない
  E_COMPLETE_HALF_EYE, // 完全に欠け眼(8近傍に打って1眼にできない)
  E_HALF_3_EYE,        // 欠け眼であるが, 3手で1眼にできる
  E_HALF_2_EYE,        // 欠け眼であるが, 2手で1眼にできる
  E_HALF_1_EYE,        // 欠け眼であるが, 1手で1眼にできる
  E_COMPLETE_ONE_EYE,  // 完全な1眼
  E_MAX,
};

// 着手を記録する構造体
typedef struct move {
    int color;  // 着手した石の色
    int pos;    // 着手箇所の座標
} move_t;


// 連を表す構造体 (19x19 : 1987bytes)
typedef struct {
    char color;                    // 連の色
    int libs;                      // 連の持つ呼吸点数
    short lib[STRING_LIB_MAX];     // 連の持つ呼吸点の座標
    int neighbors;                 // 隣接する敵の連の数
    short neighbor[MAX_NEIGHBOR];  // 隣接する敵の連の連番号
    int origin;                    // 連の始点の座標
    int size;                      // 連を構成する石の数
    bool flag;                     // 連の存在フラグ
} string_t;


typedef struct {
    char current_color; 
    struct move record[MAX_RECORDS];    // 着手箇所と色の記録
    int moves;                          // 着手数の記録
    int prisoner[S_MAX];                // アゲハマ
    int ko_pos;                         // 劫となっている箇所
    int ko_move;                        // 劫となった時の着手数

    unsigned long long current_hash;     // 現在の局面のハッシュ値
    unsigned long long previous1_hash;   // 1手前の局面のハッシュ値
    unsigned long long previous2_hash;   // 2手前の局面のハッシュ値

    char board[BOARD_MAX];              // 盤面 
    int birth_move[BOARD_MAX];          // 打たれた着手数

    int pass_count;                   // パスした回数

    pattern_t pat[BOARD_MAX];    // 周囲の石の配置 

    string_t string[MAX_STRING];        // 連のデータ(19x19 : 573,845bytes)
    int string_id[STRING_POS_MAX];    // 各座標の連のID
    int string_next[STRING_POS_MAX];  // 連を構成する石のデータ構造

    int candidates[BOARD_MAX];  // 候補手かどうかのフラグ 
//  bool seki[BOARD_MAX];
  
//  unsigned int tactical_features1[BOARD_MAX];  // 戦術的特徴 
//  unsigned int tactical_features2[BOARD_MAX];  // 戦術的特徴 

    int capture_num[S_OB];                   // 前の着手で打ち上げた石の数
//    int capture_pos[S_OB][PURE_BOARD_MAX];   // 前の着手で石を打ち上げた座標 

//  int update_num[S_OB];                    // 戦術的特徴が更新された数
//  int update_pos[S_OB][PURE_BOARD_MAX];    // 戦術的特徴が更新された座標 

//  long long rate[2][BOARD_MAX];           // シミュレーション時の各座標のレート 
//  long long sum_rate_row[2][BOARD_SIZE];  // シミュレーション時の各列のレートの合計値  
//  long long sum_rate[2];                  // シミュレーション時の全体のレートの合計値
    bool rollout;
} game_state_t;

//int onboard_pos[PURE_BOARD_MAX];
//int board_x[BOARD_MAX];
//int board_y[BOARD_MAX];

//unsigned char eye[PAT3_MAX];
//unsigned char false_eye[PAT3_MAX];
//unsigned char territory[PAT3_MAX];
//unsigned char nb4_empty[PAT3_MAX];
//unsigned char eye_condition[PAT3_MAX];


/***********
 * Point.h *
 ***********/
#define GOGUI_X(pos) (gogui_x[CORRECT_X(pos)])
#define GOGUI_Y(pos, pure_board_size) (pure_board_size + 1 - CORRECT_Y(pos))

const char gogui_x[] = { 
  'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
  'U', 'V', 'W', 'X', 'Y', 'Z' 
};
