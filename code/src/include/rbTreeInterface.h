#ifndef RBTREE_INTERFACE
#define RBTREE_INTERFACE

#include "macros.h"
#include "config.h"
#include <string.h>

struct rb_node {
	unsigned long  __rb_parent_color;
	struct rb_node *rb_right;
	struct rb_node *rb_left;
};
struct rb_root {
	struct rb_node *rb_node;
};
struct rb_root_cached {
	struct rb_root rb_root;
	struct rb_node *rb_leftmost;
};
typedef struct rb_root_cached rbRoot;
typedef struct{
    idx_t key;
    struct rb_node rb;

    /* following fields used for testing augmented RBTREE_INTERFACE functionality
    u32 val;
    u32 augmented;  ///only for AUGMENTED_TEST
    */
} rbNode;

#define RB_ROOT_CACHED (struct rb_root_cached) { {NULL, }, NULL }

int rbInsertNewKey(rbRoot *root,rbNode *node, idx_t key);
struct rb_node *rb_first(const struct rb_root *);
struct rb_node *rb_next(const struct rb_node *);
void cleanRbNodes(rbRoot* root,rbNode* nodes,idx_t nodesNum);


#endif	//RBTREE_INTERFACE
