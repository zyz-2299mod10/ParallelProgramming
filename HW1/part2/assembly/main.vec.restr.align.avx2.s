	.text
	.file	"main.c"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function main
.LCPI0_0:
	.long	0x40800000                      # float 4
.LCPI0_1:
	.long	0x30000000                      # float 4.65661287E-10
.LCPI0_2:
	.long	0xbf800000                      # float -1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3
.LCPI0_3:
	.quad	0x4010000000000000              # double 4
.LCPI0_4:
	.quad	0x41dfffffffc00000              # double 2147483647
.LCPI0_5:
	.quad	0xbff0000000000000              # double -1
.LCPI0_6:
	.quad	0x3e112e0be826d695              # double 1.0000000000000001E-9
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%rsi, %r14
	movl	%edi, %r15d
	movl	$1024, %ebx                     # imm = 0x400
	movl	$1, %r12d
	.p2align	4, 0x90
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	movl	$.L.str.3, %edx
	movl	$main.long_options, %ecx
	movl	%r15d, %edi
	movq	%r14, %rsi
	xorl	%r8d, %r8d
	callq	getopt_long
	cmpl	$115, %eax
	jne	.LBB0_2
# %bb.6:                                #   in Loop: Header=BB0_1 Depth=1
	movq	optarg(%rip), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movq	%rax, %rbx
	testl	%ebx, %ebx
	jg	.LBB0_1
	jmp	.LBB0_7
.LBB0_2:                                #   in Loop: Header=BB0_1 Depth=1
	cmpl	$-1, %eax
	je	.LBB0_10
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	cmpl	$116, %eax
	jne	.LBB0_9
# %bb.4:                                #   in Loop: Header=BB0_1 Depth=1
	movq	optarg(%rip), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movq	%rax, %r12
	addl	$-1, %eax
	cmpl	$3, %eax
	jb	.LBB0_1
# %bb.5:
	movl	$.L.str.5, %edi
	movl	%r12d, %esi
	jmp	.LBB0_8
.LBB0_7:
	movl	$.L.str.4, %edi
	movl	%ebx, %esi
.LBB0_8:
	xorl	%eax, %eax
	callq	printf
	movl	$-1, %r14d
	jmp	.LBB0_22
.LBB0_10:
	movq	%r12, -88(%rbp)                 # 8-byte Spill
	movslq	%ebx, %rax
	movq	%rsp, %rdx
	leaq	15(,%rax,4), %rcx
	andq	$-16, %rcx
	subq	%rcx, %rdx
	movq	%rdx, -48(%rbp)                 # 8-byte Spill
	movq	%rdx, %rsp
	movq	%rsp, %rdx
	subq	%rcx, %rdx
	movq	%rdx, -56(%rbp)                 # 8-byte Spill
	movq	%rdx, %rsp
	movq	%rsp, %r14
	leaq	15(,%rax,8), %rdx
	andq	$-16, %rdx
	subq	%rdx, %r14
	movq	%r14, %rsp
	movq	%rsp, %r13
	subq	%rcx, %r13
	movq	%r13, %rsp
	testl	%eax, %eax
	je	.LBB0_13
# %bb.11:
	movl	%ebx, %r15d
	xorl	%r12d, %r12d
	.p2align	4, 0x90
.LBB0_12:                               # =>This Inner Loop Header: Depth=1
	callq	rand
	vcvtsi2ss	%eax, %xmm2, %xmm0
	vmovss	.LCPI0_0(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vmulss	%xmm1, %xmm0, %xmm0
	vmovss	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vmulss	%xmm1, %xmm0, %xmm0
	vmovss	.LCPI0_2(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vaddss	%xmm1, %xmm0, %xmm0
	movq	-48(%rbp), %rax                 # 8-byte Reload
	vmovss	%xmm0, (%rax,%r12,4)
	callq	rand
	vcvtsi2ss	%eax, %xmm2, %xmm0
	vmulss	.LCPI0_0(%rip), %xmm0, %xmm0
	vmulss	.LCPI0_1(%rip), %xmm0, %xmm0
	vaddss	.LCPI0_2(%rip), %xmm0, %xmm0
	movq	-56(%rbp), %rax                 # 8-byte Reload
	vmovss	%xmm0, (%rax,%r12,4)
	callq	rand
	vcvtsi2sd	%eax, %xmm2, %xmm0
	vmulsd	.LCPI0_3(%rip), %xmm0, %xmm0
	vdivsd	.LCPI0_4(%rip), %xmm0, %xmm0
	vaddsd	.LCPI0_5(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, (%r14,%r12,8)
	movl	$0, (%r13,%r12,4)
	addq	$1, %r12
	cmpq	%r12, %r15
	jne	.LBB0_12
.LBB0_13:
	movq	%r13, -64(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %r12                 # 8-byte Reload
	movl	$.L.str.6, %edi
	movq	-88(%rbp), %r15                 # 8-byte Reload
	movl	%r15d, %esi
	xorl	%eax, %eax
	callq	printf
	leaq	-80(%rbp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_23
# %bb.14:
	movq	-80(%rbp), %rax
	movq	-72(%rbp), %r13
	cmpl	$3, %r15d
	je	.LBB0_19
# %bb.15:
	movq	%rax, %r14
	cmpl	$2, %r15d
	je	.LBB0_18
# %bb.16:
	cmpl	$1, %r15d
	movq	-56(%rbp), %rsi                 # 8-byte Reload
	movq	-64(%rbp), %rdx                 # 8-byte Reload
	jne	.LBB0_20
# %bb.17:
	movq	%r12, %rdi
	movl	%ebx, %ecx
	callq	test1
	jmp	.LBB0_20
.LBB0_9:
	movq	(%r14), %rsi
	movl	$.L.str.9, %edi
	xorl	%eax, %eax
	callq	printf
	movl	$.Lstr, %edi
	callq	puts
	movl	$.Lstr.16, %edi
	callq	puts
	movl	$.Lstr.17, %edi
	callq	puts
	movl	$.Lstr.18, %edi
	callq	puts
	movl	$1, %r14d
	jmp	.LBB0_22
.LBB0_18:
	movq	%r12, %rdi
	movq	-56(%rbp), %rsi                 # 8-byte Reload
	movq	-64(%rbp), %rdx                 # 8-byte Reload
	movl	%ebx, %ecx
	callq	test2
	jmp	.LBB0_20
.LBB0_19:
	movq	%r14, %rdi
	movq	%rax, %r14
	movl	%ebx, %esi
	callq	test3
.LBB0_20:
	leaq	-80(%rbp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_23
# %bb.21:
	movq	-80(%rbp), %rax
	subq	%r14, %rax
	movq	-72(%rbp), %rcx
	subq	%r13, %rcx
	vcvtsi2sd	%rax, %xmm2, %xmm0
	vcvtsi2sd	%rcx, %xmm2, %xmm1
	vmulsd	.LCPI0_6(%rip), %xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, -48(%rbp)                # 8-byte Spill
	xorl	%r14d, %r14d
	movl	$.L.str.7, %edi
	movl	%r15d, %esi
	xorl	%eax, %eax
	callq	printf
	movl	$.L.str.8, %edi
	vmovsd	-48(%rbp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movl	%ebx, %esi
	movl	$20000000, %edx                 # imm = 0x1312D00
	movb	$1, %al
	callq	printf
.LBB0_22:
	movl	%r14d, %eax
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB0_23:
	.cfi_def_cfa %rbp, 16
	movl	$.L.str.14, %edi
	movl	$.L.str.15, %esi
	movl	$.L__PRETTY_FUNCTION__.gettime, %ecx
	movl	$75, %edx
	callq	__assert_fail
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.globl	usage                           # -- Begin function usage
	.p2align	4, 0x90
	.type	usage,@function
usage:                                  # @usage
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rdi, %rsi
	movl	$.L.str.9, %edi
	xorl	%eax, %eax
	callq	printf
	movl	$.Lstr, %edi
	callq	puts
	movl	$.Lstr.16, %edi
	callq	puts
	movl	$.Lstr.17, %edi
	callq	puts
	movl	$.Lstr.18, %edi
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	puts                            # TAILCALL
.Lfunc_end1:
	.size	usage, .Lfunc_end1-usage
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function initValue
.LCPI2_0:
	.long	0x40800000                      # float 4
.LCPI2_1:
	.long	0x30000000                      # float 4.65661287E-10
.LCPI2_2:
	.long	0xbf800000                      # float -1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3
.LCPI2_3:
	.quad	0x4010000000000000              # double 4
.LCPI2_4:
	.quad	0x41dfffffffc00000              # double 2147483647
.LCPI2_5:
	.quad	0xbff0000000000000              # double -1
	.text
	.globl	initValue
	.p2align	4, 0x90
	.type	initValue,@function
initValue:                              # @initValue
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	testl	%r8d, %r8d
	je	.LBB2_3
# %bb.1:
	movq	%rcx, %r14
	movq	%rdx, %r15
	movq	%rsi, %r12
	movq	%rdi, %r13
	movl	%r8d, %ebx
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB2_2:                                # =>This Inner Loop Header: Depth=1
	callq	rand
	vcvtsi2ss	%eax, %xmm2, %xmm0
	vmovss	.LCPI2_0(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vmulss	%xmm1, %xmm0, %xmm0
	vmovss	.LCPI2_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vmulss	%xmm1, %xmm0, %xmm0
	vmovss	.LCPI2_2(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, (%r13,%rbp,4)
	callq	rand
	vcvtsi2ss	%eax, %xmm2, %xmm0
	vmulss	.LCPI2_0(%rip), %xmm0, %xmm0
	vmulss	.LCPI2_1(%rip), %xmm0, %xmm0
	vaddss	.LCPI2_2(%rip), %xmm0, %xmm0
	vmovss	%xmm0, (%r12,%rbp,4)
	callq	rand
	vcvtsi2sd	%eax, %xmm2, %xmm0
	vmulsd	.LCPI2_3(%rip), %xmm0, %xmm0
	vdivsd	.LCPI2_4(%rip), %xmm0, %xmm0
	vaddsd	.LCPI2_5(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, (%r15,%rbp,8)
	movl	$0, (%r14,%rbp,4)
	addq	$1, %rbp
	cmpq	%rbp, %rbx
	jne	.LBB2_2
.LBB2_3:
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	initValue, .Lfunc_end2-initValue
	.cfi_endproc
                                        # -- End function
	.type	main.long_options,@object       # @main.long_options
	.data
	.p2align	4
main.long_options:
	.quad	.L.str
	.long	1                               # 0x1
	.zero	4
	.quad	0
	.long	115                             # 0x73
	.zero	4
	.quad	.L.str.1
	.long	1                               # 0x1
	.zero	4
	.quad	0
	.long	116                             # 0x74
	.zero	4
	.quad	.L.str.2
	.long	0                               # 0x0
	.zero	4
	.quad	0
	.long	63                              # 0x3f
	.zero	4
	.zero	32
	.size	main.long_options, 128

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"size"
	.size	.L.str, 5

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"test"
	.size	.L.str.1, 5

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"help"
	.size	.L.str.2, 5

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"st:?"
	.size	.L.str.3, 5

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"Error: Workload size is set to %d (<0).\n"
	.size	.L.str.4, 41

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"Error: test%d() is not available.\n"
	.size	.L.str.5, 35

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"Running test%d()...\n"
	.size	.L.str.6, 21

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"Elapsed execution time of the loop in test%d():\n"
	.size	.L.str.7, 49

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"%lfsec (N: %d, I: %d)\n"
	.size	.L.str.8, 23

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"Usage: %s [options]\n"
	.size	.L.str.9, 21

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"r == 0"
	.size	.L.str.14, 7

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"./fasttime.h"
	.size	.L.str.15, 13

	.type	.L__PRETTY_FUNCTION__.gettime,@object # @__PRETTY_FUNCTION__.gettime
.L__PRETTY_FUNCTION__.gettime:
	.asciz	"fasttime_t gettime(void)"
	.size	.L__PRETTY_FUNCTION__.gettime, 25

	.type	.Lstr,@object                   # @str
.Lstr:
	.asciz	"Program Options:"
	.size	.Lstr, 17

	.type	.Lstr.16,@object                # @str.16
.Lstr.16:
	.asciz	"  -s  --size <N>     Use workload size N (Default = 1024)"
	.size	.Lstr.16, 58

	.type	.Lstr.17,@object                # @str.17
.Lstr.17:
	.asciz	"  -t  --test <N>     Just run the testN function (Default = 1)"
	.size	.Lstr.17, 63

	.type	.Lstr.18,@object                # @str.18
.Lstr.18:
	.asciz	"  -h  --help         This message"
	.size	.Lstr.18, 34

	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym main.long_options
