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

#define N   (-board_size)
#define S   (board_size)
#define E   (1)
#define W   (-1)
#define NN  (N+N)
#define NE  (N+E)
#define NW  (N+W)
#define SS  (S+S)
#define SE  (S+E)
#define SW  (S+W)
#define WW  (W+W)
#define EE  (E+E)


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

enum eye_condition {
  E_NOT_EYE,           // 眼でない
  E_COMPLETE_HALF_EYE, // 完全に欠け眼(8近傍に打って1眼にできない)
  E_HALF_3_EYE,        // 欠け眼であるが, 3手で1眼にできる
  E_HALF_2_EYE,        // 欠け眼であるが, 2手で1眼にできる
  E_HALF_1_EYE,        // 欠け眼であるが, 1手で1眼にできる
  E_COMPLETE_ONE_EYE,  // 完全な1眼
  E_MAX,
};

typedef struct pattern {
  unsigned int list[MD_MAX];
  unsigned long long large_list[MD_LARGE_MAX];
} pattern_t;


unsigned char eye[PAT3_MAX];
unsigned char false_eye[PAT3_MAX];
unsigned char territory[PAT3_MAX];
unsigned char nb4_empty[PAT3_MAX];
unsigned char eye_condition[PAT3_MAX];

