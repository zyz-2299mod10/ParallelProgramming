	.text
	.file	"test2.c"
	.globl	test2                           # -- Begin function test2
	.p2align	4, 0x90
	.type	test2,@function
test2:                                  # @test2
	.cfi_startproc
# %bb.0:
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%rsi,%rcx,4), %xmm0
	movaps	16(%rsi,%rcx,4), %xmm1
	maxps	(%rdi,%rcx,4), %xmm0
	maxps	16(%rdi,%rcx,4), %xmm1
	movups	%xmm0, (%rdx,%rcx,4)
	movups	%xmm1, 16(%rdx,%rcx,4)
	movaps	32(%rsi,%rcx,4), %xmm0
	movaps	48(%rsi,%rcx,4), %xmm1
	maxps	32(%rdi,%rcx,4), %xmm0
	maxps	48(%rdi,%rcx,4), %xmm1
	movups	%xmm0, 32(%rdx,%rcx,4)
	movups	%xmm1, 48(%rdx,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	jne	.LBB0_1
# %bb.4:
	retq
.Lfunc_end0:
	.size	test2, .Lfunc_end0-test2
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
