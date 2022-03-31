#ifndef MACROSLINUXMOCK
#define MACROSLINUXMOCK

////////////////////////// LINUX KERNEL Userspaceing RBTree//////////////////////////////
typedef unsigned int u32;
typedef unsigned int cycles_t;

#define offsetof(TYPE, MEMBER)	((size_t)&((TYPE *)0)->MEMBER)

/** 
 * container_of - cast a member of a structure out to the containing structure
 * @ptr:	the pointer to the member.
 * @type:	the type of the container struct this is embedded in.
 * @member:	the name of the member within the struct.
 *	TODO LESS_DEPENDENCIES
	removed:
		BUILD_BUG_ON_MSG(!__same_type(*(ptr), ((type *)0)->member) &&	\
				 !__same_type(*(ptr), void),			\
				 "pointer type mismatch in container_of()");	\
	
 */
#define container_of(ptr, type, member) ({				\
	void *__mptr = (void *)(ptr);					\
	((type *)(__mptr - offsetof(type, member))); })

#define NOOP(x)     x
//#define NOOP      do {} while(0)
#undef unlikely
#define unlikely(x) NOOP(x)
/*
 * Yes, this permits 64-bit accesses on 32-bit architectures. These will
 * actually be atomic in some cases (namely Armv7 + LPAE), but for others we
 * rely on the access being split into 2x32-bit accesses for a 32-bit quantity
 * (e.g. a virtual address) and a strong prevailing wind.
 */
#define compiletime_assert_rwonce_type(t)					\
	compiletime_assert(__native_word(t) || sizeof(t) == sizeof(long long),	\
		"Unsupported access size for {READ,WRITE}_ONCE().")
///TODO OVERWRITTEN FOR PORTING
#undef compiletime_assert_rwonce_type 
#define compiletime_assert_rwonce_type(t)   NOOP(t)
/*
 * Use __READ_ONCE() instead of READ_ONCE() if you do not require any
 * atomicity. Note that this may result in tears!
 */
#ifndef __READ_ONCE
#define __READ_ONCE(x)	(*(const volatile __unqual_scalar_typeof(x) *)&(x))
#endif

#define READ_ONCE(x)							\
({									\
	compiletime_assert_rwonce_type(x);				\
	__READ_ONCE(x);							\
})

#define __WRITE_ONCE(x, val)						\
do {									\
	*(volatile typeof(x) *)&(x) = (val);				\
} while (0)

/*//TODO ORIGNAL VERSION
#define WRITE_ONCE(x, val)						\
do {									\
	compiletime_assert_rwonce_type(x);				\
	__WRITE_ONCE(x, val);						\
} while (0)
*/
#define WRITE_ONCE(x,val)	x = val

/*
 * Use READ_ONCE_NOCHECK() instead of READ_ONCE() if you need to load a
 * word from memory atomically but without telling KASAN/KCSAN. This is
 * usually used by unwinding code when walking the stack of a running process.
 */
#define READ_ONCE_NOCHECK(x)						\
({									\
	compiletime_assert(sizeof(x) == sizeof(unsigned long),		\
		"Unsupported access size for READ_ONCE_NOCHECK().");	\
	(typeof(x))__read_once_word_nocheck(&(x));			\
})

#include <x86intrin.h>	//rdtsc
static inline cycles_t get_cycles(void)
{
	
	/**#ifndef CONFIG_X86_TSC
	if (!boot_cpu_has(X86_FEATURE_TSC))
		return 0;
	#endif */ ///TODO LESS_DEPENDENCIES

	return __rdtsc();
}

/**#define WARN_ON_ONCE(condition)	({				\
	static bool __section(".data.once") __warned;		\
	int __ret_warn_once = !!(condition);			\
								\
	if (unlikely(__ret_warn_once && !__warned)) {		\
		__warned = true;				\
		WARN_ON(1);					\
	}							\
	unlikely(__ret_warn_once);				\
}) */ //TODO LESS_DEPENDENCIES
#include <assert.h>
#define WARN_ON_ONCE(condition)	assert( !(condition) )

#define div_u64(a,b)	( a / b )
#define kfree(x)		free(x)

#endif //MACROSLINUXMOCK
