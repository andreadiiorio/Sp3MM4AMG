#include "rbTreeInterface.h"
#include "config.h"

int rbInsertNewKey(rbRoot *root,rbNode *node, idx_t key){
	ERRPRINT("NOT IMPLEMENTED\n");abort();return -1;
}
struct rb_node *rb_first(const struct rb_root *root){
	ERRPRINT("NOT IMPLEMENTED\n");abort();return NULL;
}
struct rb_node *rb_next(const struct rb_node *node){
	ERRPRINT("NOT IMPLEMENTED\n");abort();return NULL;
}
void cleanRbNodes(rbRoot* root,rbNode* nodes,idx_t nodesNum){
	memset(nodes,0,nodesNum * sizeof(*nodes));
	memset(root,0,sizeof(*root));
}


