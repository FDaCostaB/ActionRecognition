Њм%
Р
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
ћ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8Ц 
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_6/bias
w
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_6/bias
w
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes
:*
dtype0

Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/v/dense_6/kernel

)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel*
_output_shapes
:	*
dtype0

Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/m/dense_6/kernel

)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel*
_output_shapes
:	*
dtype0

Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_8/bias
x
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes	
:*
dtype0

Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_8/bias
x
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes	
:*
dtype0

Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel* 
_output_shapes
:
*
dtype0

Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel* 
_output_shapes
:
*
dtype0

Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_7/bias
x
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes	
:*
dtype0

Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_7/bias
x
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes	
:*
dtype0

Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/v/dense_7/kernel

)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes
:	*
dtype0

Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/m/dense_7/kernel

)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes
:	*
dtype0

!Adam/v/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_4/beta

5Adam/v/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_4/beta*
_output_shapes
:*
dtype0

!Adam/m/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_4/beta

5Adam/m/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_4/beta*
_output_shapes
:*
dtype0

"Adam/v/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_4/gamma

6Adam/v/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_4/gamma*
_output_shapes
:*
dtype0

"Adam/m/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_4/gamma

6Adam/m/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_4/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_10/bias
{
)Adam/v/conv2d_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_10/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_10/bias
{
)Adam/m/conv2d_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_10/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_10/kernel

+Adam/v/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_10/kernel*&
_output_shapes
: *
dtype0

Adam/m/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_10/kernel

+Adam/m/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_10/kernel*&
_output_shapes
: *
dtype0

!Adam/v/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_3/beta

5Adam/v/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_3/beta*
_output_shapes
: *
dtype0

!Adam/m/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_3/beta

5Adam/m/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_3/beta*
_output_shapes
: *
dtype0

"Adam/v/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_3/gamma

6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_3/gamma*
_output_shapes
: *
dtype0

"Adam/m/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_3/gamma

6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_3/gamma*
_output_shapes
: *
dtype0

Adam/v/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_9/bias
y
(Adam/v/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/bias*
_output_shapes
: *
dtype0

Adam/m/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_9/bias
y
(Adam/m/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/bias*
_output_shapes
: *
dtype0

Adam/v/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/v/conv2d_9/kernel

*Adam/v/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/kernel*&
_output_shapes
:  *
dtype0

Adam/m/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/m/conv2d_9/kernel

*Adam/m/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/kernel*&
_output_shapes
:  *
dtype0

!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_2/beta

5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes
: *
dtype0

!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_2/beta

5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes
: *
dtype0

"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_2/gamma

6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes
: *
dtype0

"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_2/gamma

6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
_output_shapes
: *
dtype0

Adam/v/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_8/bias
y
(Adam/v/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/bias*
_output_shapes
: *
dtype0

Adam/m/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_8/bias
y
(Adam/m/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/bias*
_output_shapes
: *
dtype0

Adam/v/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_8/kernel

*Adam/v/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/kernel*&
_output_shapes
: *
dtype0

Adam/m/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_8/kernel

*Adam/m/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/kernel*&
_output_shapes
: *
dtype0

!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_1/beta

5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
:*
dtype0

!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_1/beta

5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
:*
dtype0

"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_1/gamma

6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
:*
dtype0

"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_1/gamma

6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_7/bias
y
(Adam/v/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_7/bias
y
(Adam/m/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_7/kernel

*Adam/v/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_7/kernel

*Adam/m/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/kernel*&
_output_shapes
:*
dtype0

Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/batch_normalization/beta

3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
:*
dtype0

Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/batch_normalization/beta

3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
:*
dtype0

 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/batch_normalization/gamma

4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
:*
dtype0

 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/batch_normalization/gamma

4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
:*
dtype0

Adam/v/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_6/bias
y
(Adam/v/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_6/bias
y
(Adam/m/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_6/kernel

*Adam/v/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_6/kernel

*Adam/m/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	*
dtype0
Ђ
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
Ђ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:  *
dtype0
Ђ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
: *
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0

serving_default_input_3Placeholder*/
_output_shapes
:џџџџџџџџџpp*
dtype0*$
shape:џџџџџџџџџpp
ќ	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_6/kernelconv2d_6/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_6/kerneldense_6/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_49290

NoOpNoOp
Еи
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*яз
valueфзBрз Bиз
ѕ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer_with_weights-12
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures*
Ш
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op*
е
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2axis
	3gamma
4beta
5moving_mean
6moving_variance*

7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
Ш
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op*
е
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance*

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 

]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
Ш
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias
 k_jit_compiled_convolution_op*
е
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance*

w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses* 

}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses
Ѓkernel
	Єbias
!Ѕ_jit_compiled_convolution_op*
р
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses
	Ќaxis

­gamma
	Ўbeta
Џmoving_mean
Аmoving_variance*

Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses* 

З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses* 

Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses* 
Ў
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щkernel
	Ъbias*
Ќ
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
б_random_generator* 
Ў
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
иkernel
	йbias*
Ќ
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses
р_random_generator* 
Ў
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
чkernel
	шbias*
Ќ
)0
*1
32
43
54
65
I6
J7
S8
T9
U10
V11
i12
j13
s14
t15
u16
v17
18
19
20
21
22
23
Ѓ24
Є25
­26
Ў27
Џ28
А29
Щ30
Ъ31
и32
й33
ч34
ш35*
и
)0
*1
32
43
I4
J5
S6
T7
i8
j9
s10
t11
12
13
14
15
Ѓ16
Є17
­18
Ў19
Щ20
Ъ21
и22
й23
ч24
ш25*
* 
Е
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

юtrace_0
яtrace_1* 

№trace_0
ёtrace_1* 
* 

ђ
_variables
ѓ_iterations
є_learning_rate
ѕ_index_dict
і
_momentums
ї_velocities
ј_update_step_xla*

љserving_default* 

)0
*1*

)0
*1*
* 

њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

џtrace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
30
41
52
63*

30
41*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

I0
J1*

I0
J1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
S0
T1
U2
V3*

S0
T1*
* 

non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

Єtrace_0
Ѕtrace_1* 

Іtrace_0
Їtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

­trace_0* 

Ўtrace_0* 
* 
* 
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

Дtrace_0* 

Еtrace_0* 

i0
j1*

i0
j1*
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

Лtrace_0* 

Мtrace_0* 
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
s0
t1
u2
v3*

s0
t1*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

Тtrace_0
Уtrace_1* 

Фtrace_0
Хtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

Ыtrace_0* 

Ьtrace_0* 
* 
* 
* 

Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

вtrace_0* 

гtrace_0* 

0
1*

0
1*
* 

дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

йtrace_0* 

кtrace_0* 
_Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
0
1
2
3*

0
1*
* 

лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

рtrace_0
сtrace_1* 

тtrace_0
уtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 

Ѓ0
Є1*

Ѓ0
Є1*
* 

ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses*

№trace_0* 

ёtrace_0* 
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
­0
Ў1
Џ2
А3*

­0
Ў1*
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses*

їtrace_0
јtrace_1* 

љtrace_0
њtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

Щ0
Ъ1*

Щ0
Ъ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_7/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

и0
й1*

и0
й1*
* 

 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses*

Ѕtrace_0* 

Іtrace_0* 
_Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_8/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses* 

Ќtrace_0
­trace_1* 

Ўtrace_0
Џtrace_1* 
* 

ч0
ш1*

ч0
ш1*
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
_Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
N
50
61
U2
V3
u4
v5
6
7
Џ8
А9*
Т
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*

З0
И1*
* 
* 
* 
* 
* 
* 
з
ѓ0
Й1
К2
Л3
М4
Н5
О6
П7
Р8
С9
Т10
У11
Ф12
Х13
Ц14
Ч15
Ш16
Щ17
Ъ18
Ы19
Ь20
Э21
Ю22
Я23
а24
б25
в26
г27
д28
е29
ж30
з31
и32
й33
к34
л35
м36
н37
о38
п39
р40
с41
т42
у43
ф44
х45
ц46
ч47
ш48
щ49
ъ50
ы51
ь52*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
ф
Й0
Л1
Н2
П3
С4
У5
Х6
Ч7
Щ8
Ы9
Э10
Я11
б12
г13
е14
з15
й16
л17
н18
п19
с20
у21
х22
ч23
щ24
ы25*
ф
К0
М1
О2
Р3
Т4
Ф5
Ц6
Ш7
Ъ8
Ь9
Ю10
а11
в12
д13
ж14
и15
к16
м17
о18
р19
т20
ф21
ц22
ш23
ъ24
ь25*
* 
* 
* 
* 
* 
* 
* 
* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

u0
v1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Џ0
А1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
э	variables
ю	keras_api

яtotal

№count*
M
ё	variables
ђ	keras_api

ѓtotal

єcount
ѕ
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/conv2d_6/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_6/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_6/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_6/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_7/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_7/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_7/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_7/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_8/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_8/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_8/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_8/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_9/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_9/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_9/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_9/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_10/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_10/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_10/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_10/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_4/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_4/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_4/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_4/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_7/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_7/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_7/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_6/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_6/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_6/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_6/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*

я0
№1*

э	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ѓ0
є1*

ё	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_6/kerneldense_6/bias	iterationlearning_rateAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/conv2d_10/kernelAdam/v/conv2d_10/kernelAdam/m/conv2d_10/biasAdam/v/conv2d_10/bias"Adam/m/batch_normalization_4/gamma"Adam/v/batch_normalization_4/gamma!Adam/m/batch_normalization_4/beta!Adam/v/batch_normalization_4/betaAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biastotal_1count_1totalcountConst*k
Tind
b2`*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_51569
Ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_6/kerneldense_6/bias	iterationlearning_rateAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/conv2d_10/kernelAdam/v/conv2d_10/kernelAdam/m/conv2d_10/biasAdam/v/conv2d_10/bias"Adam/m/batch_normalization_4/gamma"Adam/v/batch_normalization_4/gamma!Adam/m/batch_normalization_4/beta!Adam/v/batch_normalization_4/betaAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biastotal_1count_1totalcount*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_51860ћЋ
­
E
)__inference_flatten_2_layer_call_fn_49875

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48730`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

П
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49598

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_48272

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѕ
Я
,__inference_sequential_2_layer_call_fn_49011
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*<
_read_only_resource_inputs
	
 !"#$*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_48823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:џџџџџџџџџpp: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%$!

_user_specified_name49007:%#!

_user_specified_name49005:%"!

_user_specified_name49003:%!!

_user_specified_name49001:% !

_user_specified_name48999:%!

_user_specified_name48997:%!

_user_specified_name48995:%!

_user_specified_name48993:%!

_user_specified_name48991:%!

_user_specified_name48989:%!

_user_specified_name48987:%!

_user_specified_name48985:%!

_user_specified_name48983:%!

_user_specified_name48981:%!

_user_specified_name48979:%!

_user_specified_name48977:%!

_user_specified_name48975:%!

_user_specified_name48973:%!

_user_specified_name48971:%!

_user_specified_name48969:%!

_user_specified_name48967:%!

_user_specified_name48965:%!

_user_specified_name48963:%!

_user_specified_name48961:%!

_user_specified_name48959:%!

_user_specified_name48957:%
!

_user_specified_name48955:%	!

_user_specified_name48953:%!

_user_specified_name48951:%!

_user_specified_name48949:%!

_user_specified_name48947:%!

_user_specified_name48945:%!

_user_specified_name48943:%!

_user_specified_name48941:%!

_user_specified_name48939:%!

_user_specified_name48937:X T
/
_output_shapes
:џџџџџџџџџpp
!
_user_specified_name	input_3
њ
ў
C__inference_conv2d_7_layer_call_and_return_conditional_losses_49436

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Ы
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49427*L
_output_shapes:
8:џџџџџџџџџ..:џџџџџџџџџ..: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ..S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs

П
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48429

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48313

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_7_layer_call_fn_49521

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_48272
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ы

'__inference_dense_6_layer_call_fn_50000

inputs
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_48816o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49996:%!

_user_specified_name49994:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э

N__inference_batch_normalization_layer_call_and_return_conditional_losses_48169

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П
F
*__inference_activation_layer_call_fn_49385

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_48531h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџff"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџff:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
њ
ў
C__inference_conv2d_9_layer_call_and_return_conditional_losses_48648

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ы
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48639*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ		 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ		 
 
_user_specified_nameinputs
Ќ
Ы
"__inference_internal_grad_fn_51093
result_grads_0
result_grads_1
result_grads_2#
mul_sequential_2_conv2d_10_beta&
"mul_sequential_2_conv2d_10_biasadd
identity

identity_1
mulMulmul_sequential_2_conv2d_10_beta"mul_sequential_2_conv2d_10_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
mul_1Mulmul_sequential_2_conv2d_10_beta"mul_sequential_2_conv2d_10_biasadd*
T0*/
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџn
SquareSquare"mul_sequential_2_conv2d_10_biasadd*
T0*/
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:ok
/
_output_shapes
:џџџџџџџџџ
8
_user_specified_name sequential_2/conv2d_10/BiasAdd:SO

_output_shapes
: 
5
_user_specified_namesequential_2/conv2d_10/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
Ѓ
Щ
"__inference_internal_grad_fn_50931
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_2_conv2d_7_beta%
!mul_sequential_2_conv2d_7_biasadd
identity

identity_1
mulMulmul_sequential_2_conv2d_7_beta!mul_sequential_2_conv2d_7_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..
mul_1Mulmul_sequential_2_conv2d_7_beta!mul_sequential_2_conv2d_7_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ..J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..m
SquareSquare!mul_sequential_2_conv2d_7_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ..b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ..:џџџџџџџџџ..: : :џџџџџџџџџ..:nj
/
_output_shapes
:џџџџџџџџџ..
7
_user_specified_namesequential_2/conv2d_7/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_2/conv2d_7/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_0

e
G__inference_activation_3_layer_call_and_return_conditional_losses_48675

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ [
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ С
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48666*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Є
у
"__inference_internal_grad_fn_51120
result_grads_0
result_grads_1
result_grads_2&
"mul_sequential_2_activation_4_beta;
7mul_sequential_2_batch_normalization_4_fusedbatchnormv3
identity

identity_1В
mulMul"mul_sequential_2_activation_4_beta7mul_sequential_2_batch_normalization_4_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџЃ
mul_1Mul"mul_sequential_2_activation_4_beta7mul_sequential_2_batch_normalization_4_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
SquareSquare7mul_sequential_2_batch_normalization_4_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:
/
_output_shapes
:џџџџџџџџџ
M
_user_specified_name53sequential_2/batch_normalization_4/FusedBatchNormV3:VR

_output_shapes
: 
8
_user_specified_name sequential_2/activation_4/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
ѕ
ї
B__inference_dense_7_layer_call_and_return_conditional_losses_49909

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџН
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49900*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_48915

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
у
"__inference_internal_grad_fn_50958
result_grads_0
result_grads_1
result_grads_2&
"mul_sequential_2_activation_1_beta;
7mul_sequential_2_batch_normalization_1_fusedbatchnormv3
identity

identity_1В
mulMul"mul_sequential_2_activation_1_beta7mul_sequential_2_batch_normalization_1_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Ѓ
mul_1Mul"mul_sequential_2_activation_1_beta7mul_sequential_2_batch_normalization_1_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџ..J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..
SquareSquare7mul_sequential_2_batch_normalization_1_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџ..b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ..:џџџџџџџџџ..: : :џџџџџџџџџ..:
/
_output_shapes
:џџџџџџџџџ..
M
_user_specified_name53sequential_2/batch_normalization_1/FusedBatchNormV3:VR

_output_shapes
: 
8
_user_specified_name sequential_2/activation_1/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_0

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_48344

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_49991

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
Щ
"__inference_internal_grad_fn_51039
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_2_conv2d_9_beta%
!mul_sequential_2_conv2d_9_biasadd
identity

identity_1
mulMulmul_sequential_2_conv2d_9_beta!mul_sequential_2_conv2d_9_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 
mul_1Mulmul_sequential_2_conv2d_9_beta!mul_sequential_2_conv2d_9_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ m
SquareSquare!mul_sequential_2_conv2d_9_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :nj
/
_output_shapes
:џџџџџџџџџ 
7
_user_specified_namesequential_2/conv2d_9/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_2/conv2d_9/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
њ
ў
C__inference_conv2d_9_layer_call_and_return_conditional_losses_49672

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ы
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49663*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ		 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ		 
 
_user_specified_nameinputs
z
щ
G__inference_sequential_2_layer_call_and_return_conditional_losses_48934
input_3(
conv2d_6_48826:
conv2d_6_48828:'
batch_normalization_48831:'
batch_normalization_48833:'
batch_normalization_48835:'
batch_normalization_48837:(
conv2d_7_48842:
conv2d_7_48844:)
batch_normalization_1_48847:)
batch_normalization_1_48849:)
batch_normalization_1_48851:)
batch_normalization_1_48853:(
conv2d_8_48858: 
conv2d_8_48860: )
batch_normalization_2_48863: )
batch_normalization_2_48865: )
batch_normalization_2_48867: )
batch_normalization_2_48869: (
conv2d_9_48874:  
conv2d_9_48876: )
batch_normalization_3_48879: )
batch_normalization_3_48881: )
batch_normalization_3_48883: )
batch_normalization_3_48885: )
conv2d_10_48889: 
conv2d_10_48891:)
batch_normalization_4_48894:)
batch_normalization_4_48896:)
batch_normalization_4_48898:)
batch_normalization_4_48900: 
dense_7_48906:	
dense_7_48908:	!
dense_8_48917:

dense_8_48919:	 
dense_6_48928:	
dense_6_48930:
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ conv2d_7/StatefulPartitionedCallЂ conv2d_8/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallі
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_6_48826conv2d_6_48828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_48504ў
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_48831batch_normalization_48833batch_normalization_48835batch_normalization_48837*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48169ё
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_48531ъ
max_pooling2d_6/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_48200
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_48842conv2d_7_48844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_48552
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_1_48847batch_normalization_1_48849batch_normalization_1_48851batch_normalization_1_48853*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48241ї
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_48579ь
max_pooling2d_7/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_48272
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_48858conv2d_8_48860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_48600
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_2_48863batch_normalization_2_48865batch_normalization_2_48867batch_normalization_2_48869*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48313ї
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_48627ь
max_pooling2d_8/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ		 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_48344
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2d_9_48874conv2d_9_48876*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_48648
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_3_48879batch_normalization_3_48881batch_normalization_3_48883batch_normalization_3_48885*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48385ї
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_48675
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_10_48889conv2d_10_48891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_48695
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_4_48894batch_normalization_4_48896batch_normalization_4_48898batch_normalization_4_48900*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48447ї
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_48722ь
max_pooling2d_9/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_48478л
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48730
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_7_48906dense_7_48908*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_48750м
dropout_4/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_48915
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_8_48917dense_8_48919*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_48787м
dropout_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_48926
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_48928dense_6_48930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_48816w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџІ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:џџџџџџџџџpp: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:%$!

_user_specified_name48930:%#!

_user_specified_name48928:%"!

_user_specified_name48919:%!!

_user_specified_name48917:% !

_user_specified_name48908:%!

_user_specified_name48906:%!

_user_specified_name48900:%!

_user_specified_name48898:%!

_user_specified_name48896:%!

_user_specified_name48894:%!

_user_specified_name48891:%!

_user_specified_name48889:%!

_user_specified_name48885:%!

_user_specified_name48883:%!

_user_specified_name48881:%!

_user_specified_name48879:%!

_user_specified_name48876:%!

_user_specified_name48874:%!

_user_specified_name48869:%!

_user_specified_name48867:%!

_user_specified_name48865:%!

_user_specified_name48863:%!

_user_specified_name48860:%!

_user_specified_name48858:%!

_user_specified_name48853:%!

_user_specified_name48851:%
!

_user_specified_name48849:%	!

_user_specified_name48847:%!

_user_specified_name48844:%!

_user_specified_name48842:%!

_user_specified_name48837:%!

_user_specified_name48835:%!

_user_specified_name48833:%!

_user_specified_name48831:%!

_user_specified_name48828:%!

_user_specified_name48826:X T
/
_output_shapes
:џџџџџџџџџpp
!
_user_specified_name	input_3
я

'__inference_dense_8_layer_call_fn_49945

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_48787p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49941:%!

_user_specified_name49939:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


(__inference_conv2d_8_layer_call_fn_49535

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_48600w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49531:%!

_user_specified_name49529:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Н
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49362

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

П
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49824

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь

'__inference_dense_7_layer_call_fn_49890

inputs
unknown:	
	unknown_0:	
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_48750p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49886:%!

_user_specified_name49884:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
ј
B__inference_dense_8_layer_call_and_return_conditional_losses_48787

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџН
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48778*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

c
E__inference_activation_layer_call_and_return_conditional_losses_48531

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff[
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџffС
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48522*L
_output_shapes:
8:џџџџџџџџџff:џџџџџџџџџff: d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџff"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџff:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs


а
5__inference_batch_normalization_1_layer_call_fn_49449

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48223
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49445:%!

_user_specified_name49443:%!

_user_specified_name49441:%!

_user_specified_name49439:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49498

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц

"__inference_internal_grad_fn_50229
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1e
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Z
mul_7Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: |
&
 _has_manual_control_dependencies(
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
ж

"__inference_internal_grad_fn_50337
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџV
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:WS
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
л

"__inference_internal_grad_fn_50418
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџW
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0


а
5__inference_batch_normalization_3_layer_call_fn_49698

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48385
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49694:%!

_user_specified_name49692:%!

_user_specified_name49690:%!

_user_specified_name49688:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_48200

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50823
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџffJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџffJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџffW
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџffb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџff^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџff^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџffE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџff:џџџџџџџџџff: : :џџџџџџџџџff:XT
/
_output_shapes
:џџџџџџџџџff
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_0
њ
ў
C__inference_conv2d_6_layer_call_and_return_conditional_losses_48504

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџffI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџffe
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџffЫ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48495*L
_output_shapes:
8:џџџџџџџџџff:џџџџџџџџџff: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџffS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџpp
 
_user_specified_nameinputs
ћ|
Б
G__inference_sequential_2_layer_call_and_return_conditional_losses_48823
input_3(
conv2d_6_48505:
conv2d_6_48507:'
batch_normalization_48510:'
batch_normalization_48512:'
batch_normalization_48514:'
batch_normalization_48516:(
conv2d_7_48553:
conv2d_7_48555:)
batch_normalization_1_48558:)
batch_normalization_1_48560:)
batch_normalization_1_48562:)
batch_normalization_1_48564:(
conv2d_8_48601: 
conv2d_8_48603: )
batch_normalization_2_48606: )
batch_normalization_2_48608: )
batch_normalization_2_48610: )
batch_normalization_2_48612: (
conv2d_9_48649:  
conv2d_9_48651: )
batch_normalization_3_48654: )
batch_normalization_3_48656: )
batch_normalization_3_48658: )
batch_normalization_3_48660: )
conv2d_10_48696: 
conv2d_10_48698:)
batch_normalization_4_48701:)
batch_normalization_4_48703:)
batch_normalization_4_48705:)
batch_normalization_4_48707: 
dense_7_48751:	
dense_7_48753:	!
dense_8_48788:

dense_8_48790:	 
dense_6_48817:	
dense_6_48819:
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ conv2d_7/StatefulPartitionedCallЂ conv2d_8/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallі
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_6_48505conv2d_6_48507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_48504ќ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_48510batch_normalization_48512batch_normalization_48514batch_normalization_48516*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48151ё
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_48531ъ
max_pooling2d_6/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_48200
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_48553conv2d_7_48555*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_48552
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_1_48558batch_normalization_1_48560batch_normalization_1_48562batch_normalization_1_48564*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48223ї
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_48579ь
max_pooling2d_7/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_48272
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_48601conv2d_8_48603*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_48600
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_2_48606batch_normalization_2_48608batch_normalization_2_48610batch_normalization_2_48612*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48295ї
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_48627ь
max_pooling2d_8/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ		 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_48344
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2d_9_48649conv2d_9_48651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_48648
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_3_48654batch_normalization_3_48656batch_normalization_3_48658batch_normalization_3_48660*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48367ї
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_48675
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_10_48696conv2d_10_48698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_48695
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_4_48701batch_normalization_4_48703batch_normalization_4_48705batch_normalization_4_48707*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48429ї
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_48722ь
max_pooling2d_9/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_48478л
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48730
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_7_48751dense_7_48753*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_48750ь
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_48767
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_8_48788dense_8_48790*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_48787
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_48804
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_48817dense_6_48819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_48816w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџю
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:џџџџџџџџџpp: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:%$!

_user_specified_name48819:%#!

_user_specified_name48817:%"!

_user_specified_name48790:%!!

_user_specified_name48788:% !

_user_specified_name48753:%!

_user_specified_name48751:%!

_user_specified_name48707:%!

_user_specified_name48705:%!

_user_specified_name48703:%!

_user_specified_name48701:%!

_user_specified_name48698:%!

_user_specified_name48696:%!

_user_specified_name48660:%!

_user_specified_name48658:%!

_user_specified_name48656:%!

_user_specified_name48654:%!

_user_specified_name48651:%!

_user_specified_name48649:%!

_user_specified_name48612:%!

_user_specified_name48610:%!

_user_specified_name48608:%!

_user_specified_name48606:%!

_user_specified_name48603:%!

_user_specified_name48601:%!

_user_specified_name48564:%!

_user_specified_name48562:%
!

_user_specified_name48560:%	!

_user_specified_name48558:%!

_user_specified_name48555:%!

_user_specified_name48553:%!

_user_specified_name48516:%!

_user_specified_name48514:%!

_user_specified_name48512:%!

_user_specified_name48510:%!

_user_specified_name48507:%!

_user_specified_name48505:X T
/
_output_shapes
:џџџџџџџџџpp
!
_user_specified_name	input_3
Ё
E
)__inference_dropout_5_layer_call_fn_49974

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_48926a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
G__inference_activation_2_layer_call_and_return_conditional_losses_49634

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ [
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ С
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49625*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


а
5__inference_batch_normalization_2_layer_call_fn_49580

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48313
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49576:%!

_user_specified_name49574:%!

_user_specified_name49572:%!

_user_specified_name49570:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ћ
џ
D__inference_conv2d_10_layer_call_and_return_conditional_losses_48695

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџe
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџЫ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48686*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ѕ
ї
B__inference_dense_7_layer_call_and_return_conditional_losses_48750

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџН
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48741*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50526
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :XT
/
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
Я

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49734

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50796
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџffJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџffJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџffV
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџffb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџff^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџff^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџffE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџff:џџџџџџџџџff: : :џџџџџџџџџff:WS
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_0
Є
у
"__inference_internal_grad_fn_51066
result_grads_0
result_grads_1
result_grads_2&
"mul_sequential_2_activation_3_beta;
7mul_sequential_2_batch_normalization_3_fusedbatchnormv3
identity

identity_1В
mulMul"mul_sequential_2_activation_3_beta7mul_sequential_2_batch_normalization_3_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ѓ
mul_1Mul"mul_sequential_2_activation_3_beta7mul_sequential_2_batch_normalization_3_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 
SquareSquare7mul_sequential_2_batch_normalization_3_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :
/
_output_shapes
:џџџџџџџџџ 
M
_user_specified_name53sequential_2/batch_normalization_3/FusedBatchNormV3:VR

_output_shapes
: 
8
_user_specified_name sequential_2/activation_3/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
Ф
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_48730

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

П
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49716

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
њ
ў
C__inference_conv2d_7_layer_call_and_return_conditional_losses_48552

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Ы
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48543*L
_output_shapes:
8:џџџџџџџџџ..:џџџџџџџџџ..: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ..S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
Я

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48447

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц

"__inference_internal_grad_fn_50310
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1e
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Z
mul_7Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: |
&
 _has_manual_control_dependencies(
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
АЏ
њ%
 __inference__wrapped_model_48133
input_3N
4sequential_2_conv2d_6_conv2d_readvariableop_resource:C
5sequential_2_conv2d_6_biasadd_readvariableop_resource:F
8sequential_2_batch_normalization_readvariableop_resource:H
:sequential_2_batch_normalization_readvariableop_1_resource:W
Isequential_2_batch_normalization_fusedbatchnormv3_readvariableop_resource:Y
Ksequential_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_7_conv2d_readvariableop_resource:C
5sequential_2_conv2d_7_biasadd_readvariableop_resource:H
:sequential_2_batch_normalization_1_readvariableop_resource:J
<sequential_2_batch_normalization_1_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_8_conv2d_readvariableop_resource: C
5sequential_2_conv2d_8_biasadd_readvariableop_resource: H
:sequential_2_batch_normalization_2_readvariableop_resource: J
<sequential_2_batch_normalization_2_readvariableop_1_resource: Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource: [
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource: N
4sequential_2_conv2d_9_conv2d_readvariableop_resource:  C
5sequential_2_conv2d_9_biasadd_readvariableop_resource: H
:sequential_2_batch_normalization_3_readvariableop_resource: J
<sequential_2_batch_normalization_3_readvariableop_1_resource: Y
Ksequential_2_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: [
Msequential_2_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_2_conv2d_10_conv2d_readvariableop_resource: D
6sequential_2_conv2d_10_biasadd_readvariableop_resource:H
:sequential_2_batch_normalization_4_readvariableop_resource:J
<sequential_2_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:F
3sequential_2_dense_7_matmul_readvariableop_resource:	C
4sequential_2_dense_7_biasadd_readvariableop_resource:	G
3sequential_2_dense_8_matmul_readvariableop_resource:
C
4sequential_2_dense_8_biasadd_readvariableop_resource:	F
3sequential_2_dense_6_matmul_readvariableop_resource:	B
4sequential_2_dense_6_biasadd_readvariableop_resource:
identityЂ@sequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOpЂBsequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ/sequential_2/batch_normalization/ReadVariableOpЂ1sequential_2/batch_normalization/ReadVariableOp_1ЂBsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂDsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ1sequential_2/batch_normalization_1/ReadVariableOpЂ3sequential_2/batch_normalization_1/ReadVariableOp_1ЂBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ђ1sequential_2/batch_normalization_2/ReadVariableOpЂ3sequential_2/batch_normalization_2/ReadVariableOp_1ЂBsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂDsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ1sequential_2/batch_normalization_3/ReadVariableOpЂ3sequential_2/batch_normalization_3/ReadVariableOp_1ЂBsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЂDsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ђ1sequential_2/batch_normalization_4/ReadVariableOpЂ3sequential_2/batch_normalization_4/ReadVariableOp_1Ђ-sequential_2/conv2d_10/BiasAdd/ReadVariableOpЂ,sequential_2/conv2d_10/Conv2D/ReadVariableOpЂ,sequential_2/conv2d_6/BiasAdd/ReadVariableOpЂ+sequential_2/conv2d_6/Conv2D/ReadVariableOpЂ,sequential_2/conv2d_7/BiasAdd/ReadVariableOpЂ+sequential_2/conv2d_7/Conv2D/ReadVariableOpЂ,sequential_2/conv2d_8/BiasAdd/ReadVariableOpЂ+sequential_2/conv2d_8/Conv2D/ReadVariableOpЂ,sequential_2/conv2d_9/BiasAdd/ReadVariableOpЂ+sequential_2/conv2d_9/Conv2D/ReadVariableOpЂ+sequential_2/dense_6/BiasAdd/ReadVariableOpЂ*sequential_2/dense_6/MatMul/ReadVariableOpЂ+sequential_2/dense_7/BiasAdd/ReadVariableOpЂ*sequential_2/dense_7/MatMul/ReadVariableOpЂ+sequential_2/dense_8/BiasAdd/ReadVariableOpЂ*sequential_2/dense_8/MatMul/ReadVariableOpЈ
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ч
sequential_2/conv2d_6/Conv2DConv2Dinput_33sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides

,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff_
sequential_2/conv2d_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential_2/conv2d_6/mulMul#sequential_2/conv2d_6/beta:output:0&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџff
sequential_2/conv2d_6/SigmoidSigmoidsequential_2/conv2d_6/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџffЇ
sequential_2/conv2d_6/mul_1Mul&sequential_2/conv2d_6/BiasAdd:output:0!sequential_2/conv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff
sequential_2/conv2d_6/IdentityIdentitysequential_2/conv2d_6/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџffЃ
sequential_2/conv2d_6/IdentityN	IdentityNsequential_2/conv2d_6/mul_1:z:0&sequential_2/conv2d_6/BiasAdd:output:0#sequential_2/conv2d_6/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-47904*L
_output_shapes:
8:џџџџџџџџџff:џџџџџџџџџff: Є
/sequential_2/batch_normalization/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0Ј
1sequential_2/batch_normalization/ReadVariableOp_1ReadVariableOp:sequential_2_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0Ц
@sequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
Bsequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0§
1sequential_2/batch_normalization/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_6/IdentityN:output:07sequential_2/batch_normalization/ReadVariableOp:value:09sequential_2/batch_normalization/ReadVariableOp_1:value:0Hsequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jsequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
is_training( a
sequential_2/activation/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
sequential_2/activation/mulMul%sequential_2/activation/beta:output:05sequential_2/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџff
sequential_2/activation/SigmoidSigmoidsequential_2/activation/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџffК
sequential_2/activation/mul_1Mul5sequential_2/batch_normalization/FusedBatchNormV3:y:0#sequential_2/activation/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff
 sequential_2/activation/IdentityIdentity!sequential_2/activation/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџffИ
!sequential_2/activation/IdentityN	IdentityN!sequential_2/activation/mul_1:z:05sequential_2/batch_normalization/FusedBatchNormV3:y:0%sequential_2/activation/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-47927*L
_output_shapes:
8:џџџџџџџџџff:џџџџџџџџџff: Ш
$sequential_2/max_pooling2d_6/MaxPoolMaxPool*sequential_2/activation/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ22*
ksize
*
paddingVALID*
strides
Ј
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0э
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides

,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ.._
sequential_2/conv2d_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential_2/conv2d_7/mulMul#sequential_2/conv2d_7/beta:output:0&sequential_2/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..
sequential_2/conv2d_7/SigmoidSigmoidsequential_2/conv2d_7/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Ї
sequential_2/conv2d_7/mul_1Mul&sequential_2/conv2d_7/BiasAdd:output:0!sequential_2/conv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..
sequential_2/conv2d_7/IdentityIdentitysequential_2/conv2d_7/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Ѓ
sequential_2/conv2d_7/IdentityN	IdentityNsequential_2/conv2d_7/mul_1:z:0&sequential_2/conv2d_7/BiasAdd:output:0#sequential_2/conv2d_7/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-47943*L
_output_shapes:
8:џџџџџџџџџ..:џџџџџџџџџ..: Ј
1sequential_2/batch_normalization_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
3sequential_2/batch_normalization_1/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0Ъ
Bsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ю
Dsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
3sequential_2/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_7/IdentityN:output:09sequential_2/batch_normalization_1/ReadVariableOp:value:0;sequential_2/batch_normalization_1/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ..:::::*
epsilon%o:*
is_training( c
sequential_2/activation_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Р
sequential_2/activation_1/mulMul'sequential_2/activation_1/beta:output:07sequential_2/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..
!sequential_2/activation_1/SigmoidSigmoid!sequential_2/activation_1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Р
sequential_2/activation_1/mul_1Mul7sequential_2/batch_normalization_1/FusedBatchNormV3:y:0%sequential_2/activation_1/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..
"sequential_2/activation_1/IdentityIdentity#sequential_2/activation_1/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Р
#sequential_2/activation_1/IdentityN	IdentityN#sequential_2/activation_1/mul_1:z:07sequential_2/batch_normalization_1/FusedBatchNormV3:y:0'sequential_2/activation_1/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-47966*L
_output_shapes:
8:џџџџџџџџџ..:џџџџџџџџџ..: Ъ
$sequential_2/max_pooling2d_7/MaxPoolMaxPool,sequential_2/activation_1/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Ј
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0э
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ _
sequential_2/conv2d_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential_2/conv2d_8/mulMul#sequential_2/conv2d_8/beta:output:0&sequential_2/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_2/conv2d_8/SigmoidSigmoidsequential_2/conv2d_8/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ї
sequential_2/conv2d_8/mul_1Mul&sequential_2/conv2d_8/BiasAdd:output:0!sequential_2/conv2d_8/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_2/conv2d_8/IdentityIdentitysequential_2/conv2d_8/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ѓ
sequential_2/conv2d_8/IdentityN	IdentityNsequential_2/conv2d_8/mul_1:z:0&sequential_2/conv2d_8/BiasAdd:output:0#sequential_2/conv2d_8/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-47982*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : Ј
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype0Ќ
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype0Ъ
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ю
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_8/IdentityN:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( c
sequential_2/activation_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Р
sequential_2/activation_2/mulMul'sequential_2/activation_2/beta:output:07sequential_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
!sequential_2/activation_2/SigmoidSigmoid!sequential_2/activation_2/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Р
sequential_2/activation_2/mul_1Mul7sequential_2/batch_normalization_2/FusedBatchNormV3:y:0%sequential_2/activation_2/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
"sequential_2/activation_2/IdentityIdentity#sequential_2/activation_2/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Р
#sequential_2/activation_2/IdentityN	IdentityN#sequential_2/activation_2/mul_1:z:07sequential_2/batch_normalization_2/FusedBatchNormV3:y:0'sequential_2/activation_2/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48005*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : Ъ
$sequential_2/max_pooling2d_8/MaxPoolMaxPool,sequential_2/activation_2/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ		 *
ksize
*
paddingVALID*
strides
Ј
+sequential_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0э
sequential_2/conv2d_9/Conv2DConv2D-sequential_2/max_pooling2d_8/MaxPool:output:03sequential_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

,sequential_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
sequential_2/conv2d_9/BiasAddBiasAdd%sequential_2/conv2d_9/Conv2D:output:04sequential_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ _
sequential_2/conv2d_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential_2/conv2d_9/mulMul#sequential_2/conv2d_9/beta:output:0&sequential_2/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_2/conv2d_9/SigmoidSigmoidsequential_2/conv2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ї
sequential_2/conv2d_9/mul_1Mul&sequential_2/conv2d_9/BiasAdd:output:0!sequential_2/conv2d_9/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_2/conv2d_9/IdentityIdentitysequential_2/conv2d_9/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ѓ
sequential_2/conv2d_9/IdentityN	IdentityNsequential_2/conv2d_9/mul_1:z:0&sequential_2/conv2d_9/BiasAdd:output:0#sequential_2/conv2d_9/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48021*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : Ј
1sequential_2/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0Ќ
3sequential_2/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ъ
Bsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ю
Dsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
3sequential_2/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_9/IdentityN:output:09sequential_2/batch_normalization_3/ReadVariableOp:value:0;sequential_2/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( c
sequential_2/activation_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Р
sequential_2/activation_3/mulMul'sequential_2/activation_3/beta:output:07sequential_2/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
!sequential_2/activation_3/SigmoidSigmoid!sequential_2/activation_3/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Р
sequential_2/activation_3/mul_1Mul7sequential_2/batch_normalization_3/FusedBatchNormV3:y:0%sequential_2/activation_3/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
"sequential_2/activation_3/IdentityIdentity#sequential_2/activation_3/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Р
#sequential_2/activation_3/IdentityN	IdentityN#sequential_2/activation_3/mul_1:z:07sequential_2/batch_normalization_3/FusedBatchNormV3:y:0'sequential_2/activation_3/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48044*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : Њ
,sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ю
sequential_2/conv2d_10/Conv2DConv2D,sequential_2/activation_3/IdentityN:output:04sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
 
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
sequential_2/conv2d_10/BiasAddBiasAdd&sequential_2/conv2d_10/Conv2D:output:05sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ`
sequential_2/conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Њ
sequential_2/conv2d_10/mulMul$sequential_2/conv2d_10/beta:output:0'sequential_2/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
sequential_2/conv2d_10/SigmoidSigmoidsequential_2/conv2d_10/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџЊ
sequential_2/conv2d_10/mul_1Mul'sequential_2/conv2d_10/BiasAdd:output:0"sequential_2/conv2d_10/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
sequential_2/conv2d_10/IdentityIdentity sequential_2/conv2d_10/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџЇ
 sequential_2/conv2d_10/IdentityN	IdentityN sequential_2/conv2d_10/mul_1:z:0'sequential_2/conv2d_10/BiasAdd:output:0$sequential_2/conv2d_10/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48059*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ: Ј
1sequential_2/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
3sequential_2/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype0Ъ
Bsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ю
Dsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
3sequential_2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3)sequential_2/conv2d_10/IdentityN:output:09sequential_2/batch_normalization_4/ReadVariableOp:value:0;sequential_2/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( c
sequential_2/activation_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Р
sequential_2/activation_4/mulMul'sequential_2/activation_4/beta:output:07sequential_2/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
!sequential_2/activation_4/SigmoidSigmoid!sequential_2/activation_4/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџР
sequential_2/activation_4/mul_1Mul7sequential_2/batch_normalization_4/FusedBatchNormV3:y:0%sequential_2/activation_4/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
"sequential_2/activation_4/IdentityIdentity#sequential_2/activation_4/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџР
#sequential_2/activation_4/IdentityN	IdentityN#sequential_2/activation_4/mul_1:z:07sequential_2/batch_normalization_4/FusedBatchNormV3:y:0'sequential_2/activation_4/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48082*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ: Ъ
$sequential_2/max_pooling2d_9/MaxPoolMaxPool,sequential_2/activation_4/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
m
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Б
sequential_2/flatten_2/ReshapeReshape-sequential_2/max_pooling2d_9/MaxPool:output:0%sequential_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Е
sequential_2/dense_7/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ж
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
sequential_2/dense_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
sequential_2/dense_7/mulMul"sequential_2/dense_7/beta:output:0%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџx
sequential_2/dense_7/SigmoidSigmoidsequential_2/dense_7/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_2/dense_7/mul_1Mul%sequential_2/dense_7/BiasAdd:output:0 sequential_2/dense_7/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ|
sequential_2/dense_7/IdentityIdentitysequential_2/dense_7/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_2/dense_7/IdentityN	IdentityNsequential_2/dense_7/mul_1:z:0%sequential_2/dense_7/BiasAdd:output:0"sequential_2/dense_7/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48100*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ: 
sequential_2/dropout_4/IdentityIdentity'sequential_2/dense_7/IdentityN:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ж
sequential_2/dense_8/MatMulMatMul(sequential_2/dropout_4/Identity:output:02sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ж
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
sequential_2/dense_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
sequential_2/dense_8/mulMul"sequential_2/dense_8/beta:output:0%sequential_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџx
sequential_2/dense_8/SigmoidSigmoidsequential_2/dense_8/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_2/dense_8/mul_1Mul%sequential_2/dense_8/BiasAdd:output:0 sequential_2/dense_8/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ|
sequential_2/dense_8/IdentityIdentitysequential_2/dense_8/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_2/dense_8/IdentityN	IdentityNsequential_2/dense_8/mul_1:z:0%sequential_2/dense_8/BiasAdd:output:0"sequential_2/dense_8/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48116*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ: 
sequential_2/dropout_5/IdentityIdentity'sequential_2/dense_8/IdentityN:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Е
sequential_2/dense_6/MatMulMatMul(sequential_2/dropout_5/Identity:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_2/dense_6/SoftmaxSoftmax%sequential_2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџu
IdentityIdentity&sequential_2/dense_6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЬ
NoOpNoOpA^sequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOpC^sequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_10^sequential_2/batch_normalization/ReadVariableOp2^sequential_2/batch_normalization/ReadVariableOp_1C^sequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_1/ReadVariableOp4^sequential_2/batch_normalization_1/ReadVariableOp_1C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1C^sequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_3/ReadVariableOp4^sequential_2/batch_normalization_3/ReadVariableOp_1C^sequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_4/ReadVariableOp4^sequential_2/batch_normalization_4/ReadVariableOp_1.^sequential_2/conv2d_10/BiasAdd/ReadVariableOp-^sequential_2/conv2d_10/Conv2D/ReadVariableOp-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp-^sequential_2/conv2d_9/BiasAdd/ReadVariableOp,^sequential_2/conv2d_9/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:џџџџџџџџџpp: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Bsequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Bsequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12
@sequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp@sequential_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2f
1sequential_2/batch_normalization/ReadVariableOp_11sequential_2/batch_normalization/ReadVariableOp_12b
/sequential_2/batch_normalization/ReadVariableOp/sequential_2/batch_normalization/ReadVariableOp2
Dsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12
Bsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2j
3sequential_2/batch_normalization_1/ReadVariableOp_13sequential_2/batch_normalization_1/ReadVariableOp_12f
1sequential_2/batch_normalization_1/ReadVariableOp1sequential_2/batch_normalization_1/ReadVariableOp2
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2
Dsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12
Bsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2j
3sequential_2/batch_normalization_3/ReadVariableOp_13sequential_2/batch_normalization_3/ReadVariableOp_12f
1sequential_2/batch_normalization_3/ReadVariableOp1sequential_2/batch_normalization_3/ReadVariableOp2
Dsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12
Bsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2j
3sequential_2/batch_normalization_4/ReadVariableOp_13sequential_2/batch_normalization_4/ReadVariableOp_12f
1sequential_2/batch_normalization_4/ReadVariableOp1sequential_2/batch_normalization_4/ReadVariableOp2^
-sequential_2/conv2d_10/BiasAdd/ReadVariableOp-sequential_2/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_10/Conv2D/ReadVariableOp,sequential_2/conv2d_10/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_9/BiasAdd/ReadVariableOp,sequential_2/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_9/Conv2D/ReadVariableOp+sequential_2/conv2d_9/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
/
_output_shapes
:џџџџџџџџџpp
!
_user_specified_name	input_3


)__inference_conv2d_10_layer_call_fn_49761

inputs!
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_48695w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49757:%!

_user_specified_name49755:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50661
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ..J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..V
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ..b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ..:џџџџџџџџџ..: : :џџџџџџџџџ..:WS
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_0

f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_48478

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
G__inference_activation_4_layer_call_and_return_conditional_losses_48722

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ[
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџС
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48713*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ: d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50445
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ V
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :WS
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
л

"__inference_internal_grad_fn_50391
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџW
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
Ж
K
/__inference_max_pooling2d_9_layer_call_fn_49865

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_48478
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


а
5__inference_batch_normalization_4_layer_call_fn_49793

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48429
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49789:%!

_user_specified_name49787:%!

_user_specified_name49785:%!

_user_specified_name49783:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48241

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

П
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48367

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
У
H
,__inference_activation_2_layer_call_fn_49621

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_48627h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Я
b
)__inference_dropout_5_layer_call_fn_49969

inputs
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_48804p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50715
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ..J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ..b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ..:џџџџџџџџџ..: : :џџџџџџџџџ..:XT
/
_output_shapes
:џџџџџџџџџ..
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_0
У
H
,__inference_activation_1_layer_call_fn_49503

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_48579h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ.."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ..:W S
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs
Ц

"__inference_internal_grad_fn_50256
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1e
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Z
mul_7Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: |
&
 _has_manual_control_dependencies(
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0

c
E__inference_activation_layer_call_and_return_conditional_losses_49398

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff[
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџffС
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49389*L
_output_shapes:
8:џџџџџџџџџff:џџџџџџџџџff: d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџff"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџff:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
Я
Ц
#__inference_signature_wrapper_49290
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_48133o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:џџџџџџџџџpp: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%$!

_user_specified_name49286:%#!

_user_specified_name49284:%"!

_user_specified_name49282:%!!

_user_specified_name49280:% !

_user_specified_name49278:%!

_user_specified_name49276:%!

_user_specified_name49274:%!

_user_specified_name49272:%!

_user_specified_name49270:%!

_user_specified_name49268:%!

_user_specified_name49266:%!

_user_specified_name49264:%!

_user_specified_name49262:%!

_user_specified_name49260:%!

_user_specified_name49258:%!

_user_specified_name49256:%!

_user_specified_name49254:%!

_user_specified_name49252:%!

_user_specified_name49250:%!

_user_specified_name49248:%!

_user_specified_name49246:%!

_user_specified_name49244:%!

_user_specified_name49242:%!

_user_specified_name49240:%!

_user_specified_name49238:%!

_user_specified_name49236:%
!

_user_specified_name49234:%	!

_user_specified_name49232:%!

_user_specified_name49230:%!

_user_specified_name49228:%!

_user_specified_name49226:%!

_user_specified_name49224:%!

_user_specified_name49222:%!

_user_specified_name49220:%!

_user_specified_name49218:%!

_user_specified_name49216:X T
/
_output_shapes
:џџџџџџџџџpp
!
_user_specified_name	input_3
ж

"__inference_internal_grad_fn_50580
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ V
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :WS
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0

Ч
"__inference_internal_grad_fn_51174
result_grads_0
result_grads_1
result_grads_2!
mul_sequential_2_dense_8_beta$
 mul_sequential_2_dense_8_biasadd
identity

identity_1
mulMulmul_sequential_2_dense_8_beta mul_sequential_2_dense_8_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
mul_1Mulmul_sequential_2_dense_8_beta mul_sequential_2_dense_8_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
SquareSquare mul_sequential_2_dense_8_biasadd*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Z
mul_7Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:fb
(
_output_shapes
:џџџџџџџџџ
6
_user_specified_namesequential_2/dense_8/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namesequential_2/dense_8/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: |
&
 _has_manual_control_dependencies(
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_48804

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


(__inference_conv2d_7_layer_call_fn_49417

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_48552w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ..<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49413:%!

_user_specified_name49411:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50607
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :XT
/
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
У
H
,__inference_activation_4_layer_call_fn_49847

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_48722h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
в

є
B__inference_dense_6_layer_call_and_return_conditional_losses_50011

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я
b
)__inference_dropout_4_layer_call_fn_49914

inputs
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_48767p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
ў
C__inference_conv2d_8_layer_call_and_return_conditional_losses_49554

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ы
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49545*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_49881

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


а
5__inference_batch_normalization_2_layer_call_fn_49567

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48295
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49563:%!

_user_specified_name49561:%!

_user_specified_name49559:%!

_user_specified_name49557:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
л
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_48926

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_48767

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ
Я
,__inference_sequential_2_layer_call_fn_49088
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_48934o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:џџџџџџџџџpp: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%$!

_user_specified_name49084:%#!

_user_specified_name49082:%"!

_user_specified_name49080:%!!

_user_specified_name49078:% !

_user_specified_name49076:%!

_user_specified_name49074:%!

_user_specified_name49072:%!

_user_specified_name49070:%!

_user_specified_name49068:%!

_user_specified_name49066:%!

_user_specified_name49064:%!

_user_specified_name49062:%!

_user_specified_name49060:%!

_user_specified_name49058:%!

_user_specified_name49056:%!

_user_specified_name49054:%!

_user_specified_name49052:%!

_user_specified_name49050:%!

_user_specified_name49048:%!

_user_specified_name49046:%!

_user_specified_name49044:%!

_user_specified_name49042:%!

_user_specified_name49040:%!

_user_specified_name49038:%!

_user_specified_name49036:%!

_user_specified_name49034:%
!

_user_specified_name49032:%	!

_user_specified_name49030:%!

_user_specified_name49028:%!

_user_specified_name49026:%!

_user_specified_name49024:%!

_user_specified_name49022:%!

_user_specified_name49020:%!

_user_specified_name49018:%!

_user_specified_name49016:%!

_user_specified_name49014:X T
/
_output_shapes
:џџџџџџџџџpp
!
_user_specified_name	input_3


Ю
3__inference_batch_normalization_layer_call_fn_49344

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48169
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49340:%!

_user_specified_name49338:%!

_user_specified_name49336:%!

_user_specified_name49334:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Є
у
"__inference_internal_grad_fn_51012
result_grads_0
result_grads_1
result_grads_2&
"mul_sequential_2_activation_2_beta;
7mul_sequential_2_batch_normalization_2_fusedbatchnormv3
identity

identity_1В
mulMul"mul_sequential_2_activation_2_beta7mul_sequential_2_batch_normalization_2_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ѓ
mul_1Mul"mul_sequential_2_activation_2_beta7mul_sequential_2_batch_normalization_2_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 
SquareSquare7mul_sequential_2_batch_normalization_2_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :
/
_output_shapes
:џџџџџџџџџ 
M
_user_specified_name53sequential_2/batch_normalization_2/FusedBatchNormV3:VR

_output_shapes
: 
8
_user_specified_name sequential_2/activation_2/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
ж

"__inference_internal_grad_fn_50364
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџV
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:WS
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
љ
ј
B__inference_dense_8_layer_call_and_return_conditional_losses_49964

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџН
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49955*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
Щ
"__inference_internal_grad_fn_50985
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_2_conv2d_8_beta%
!mul_sequential_2_conv2d_8_biasadd
identity

identity_1
mulMulmul_sequential_2_conv2d_8_beta!mul_sequential_2_conv2d_8_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 
mul_1Mulmul_sequential_2_conv2d_8_beta!mul_sequential_2_conv2d_8_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ m
SquareSquare!mul_sequential_2_conv2d_8_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :nj
/
_output_shapes
:џџџџџџџџџ 
7
_user_specified_namesequential_2/conv2d_8/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_2/conv2d_8/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
л
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_49936

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50850
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџffJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџffJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџffW
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџffb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџff^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџff^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџffE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџff:џџџџџџџџџff: : :џџџџџџџџџff:XT
/
_output_shapes
:џџџџџџџџџff
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_0
Ц

"__inference_internal_grad_fn_50283
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1e
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Z
mul_7Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: |
&
 _has_manual_control_dependencies(
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
л

"__inference_internal_grad_fn_50634
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :XT
/
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
Я

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49616

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э

N__inference_batch_normalization_layer_call_and_return_conditional_losses_49380

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
ў
C__inference_conv2d_8_layer_call_and_return_conditional_losses_48600

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Ы
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48591*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
G__inference_activation_2_layer_call_and_return_conditional_losses_48627

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ [
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ С
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48618*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ё
E
)__inference_dropout_4_layer_call_fn_49919

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_48915a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_49986

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


(__inference_conv2d_9_layer_call_fn_49653

inputs!
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_48648w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ		 : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49649:%!

_user_specified_name49647:W S
/
_output_shapes
:џџџџџџџџџ		 
 
_user_specified_nameinputs

Н
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48151

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


а
5__inference_batch_normalization_4_layer_call_fn_49806

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48447
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49802:%!

_user_specified_name49800:%!

_user_specified_name49798:%!

_user_specified_name49796:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


а
5__inference_batch_normalization_1_layer_call_fn_49462

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48241
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49458:%!

_user_specified_name49456:%!

_user_specified_name49454:%!

_user_specified_name49452:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ћ
џ
D__inference_conv2d_10_layer_call_and_return_conditional_losses_49780

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџe
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџЫ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49771*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
У
H
,__inference_activation_3_layer_call_fn_49739

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_48675h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_8_layer_call_fn_49639

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_48344
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

п
"__inference_internal_grad_fn_50904
result_grads_0
result_grads_1
result_grads_2$
 mul_sequential_2_activation_beta9
5mul_sequential_2_batch_normalization_fusedbatchnormv3
identity

identity_1Ў
mulMul mul_sequential_2_activation_beta5mul_sequential_2_batch_normalization_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff
mul_1Mul mul_sequential_2_activation_beta5mul_sequential_2_batch_normalization_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџffJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџffJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџff
SquareSquare5mul_sequential_2_batch_normalization_fusedbatchnormv3*
T0*/
_output_shapes
:џџџџџџџџџffb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџff^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџff^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџffE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџff:џџџџџџџџџff: : :џџџџџџџџџff:~
/
_output_shapes
:џџџџџџџџџff
K
_user_specified_name31sequential_2/batch_normalization/FusedBatchNormV3:TP

_output_shapes
: 
6
_user_specified_namesequential_2/activation/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_0


(__inference_conv2d_6_layer_call_fn_49299

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_48504w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџff<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџpp: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49295:%!

_user_specified_name49293:W S
/
_output_shapes
:џџџџџџџџџpp
 
_user_specified_nameinputs
Ѓ
Щ
"__inference_internal_grad_fn_50877
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_2_conv2d_6_beta%
!mul_sequential_2_conv2d_6_biasadd
identity

identity_1
mulMulmul_sequential_2_conv2d_6_beta!mul_sequential_2_conv2d_6_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff
mul_1Mulmul_sequential_2_conv2d_6_beta!mul_sequential_2_conv2d_6_biasadd*
T0*/
_output_shapes
:џџџџџџџџџffJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџffJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџffm
SquareSquare!mul_sequential_2_conv2d_6_biasadd*
T0*/
_output_shapes
:џџџџџџџџџffb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџff^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџff^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџffE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџff:џџџџџџџџџff: : :џџџџџџџџџff:nj
/
_output_shapes
:џџџџџџџџџff
7
_user_specified_namesequential_2/conv2d_6/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_2/conv2d_6/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_0

e
G__inference_activation_4_layer_call_and_return_conditional_losses_49860

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ[
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџС
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49851*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ: d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_49644

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50769
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџffJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffZ
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџffJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџffV
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџffb
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџff^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџff\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџff^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџffE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџff:џџџџџџџџџff: : :џџџџџџџџџff:WS
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџff
(
_user_specified_nameresult_grads_0
њ
ў
C__inference_conv2d_6_layer_call_and_return_conditional_losses_49318

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџffI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџffU
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџffe
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџffY
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџffЫ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49309*L
_output_shapes:
8:џџџџџџџџџff:џџџџџџџџџff: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџffS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџpp
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50688
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ..J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..V
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ..b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ..:џџџџџџџџџ..: : :џџџџџџџџџ..:WS
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_0
в

є
B__inference_dense_6_layer_call_and_return_conditional_losses_48816

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_49526

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
G__inference_activation_1_layer_call_and_return_conditional_losses_48579

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..[
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..С
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-48570*L
_output_shapes:
8:џџџџџџџџџ..:џџџџџџџџџ..: d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ.."!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ..:W S
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50499
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :XT
/
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0
Я

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49842

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
яЖ
х=
!__inference__traced_restore_51860
file_prefix:
 assignvariableop_conv2d_6_kernel:.
 assignvariableop_1_conv2d_6_bias::
,assignvariableop_2_batch_normalization_gamma:9
+assignvariableop_3_batch_normalization_beta:@
2assignvariableop_4_batch_normalization_moving_mean:D
6assignvariableop_5_batch_normalization_moving_variance:<
"assignvariableop_6_conv2d_7_kernel:.
 assignvariableop_7_conv2d_7_bias:<
.assignvariableop_8_batch_normalization_1_gamma:;
-assignvariableop_9_batch_normalization_1_beta:C
5assignvariableop_10_batch_normalization_1_moving_mean:G
9assignvariableop_11_batch_normalization_1_moving_variance:=
#assignvariableop_12_conv2d_8_kernel: /
!assignvariableop_13_conv2d_8_bias: =
/assignvariableop_14_batch_normalization_2_gamma: <
.assignvariableop_15_batch_normalization_2_beta: C
5assignvariableop_16_batch_normalization_2_moving_mean: G
9assignvariableop_17_batch_normalization_2_moving_variance: =
#assignvariableop_18_conv2d_9_kernel:  /
!assignvariableop_19_conv2d_9_bias: =
/assignvariableop_20_batch_normalization_3_gamma: <
.assignvariableop_21_batch_normalization_3_beta: C
5assignvariableop_22_batch_normalization_3_moving_mean: G
9assignvariableop_23_batch_normalization_3_moving_variance: >
$assignvariableop_24_conv2d_10_kernel: 0
"assignvariableop_25_conv2d_10_bias:=
/assignvariableop_26_batch_normalization_4_gamma:<
.assignvariableop_27_batch_normalization_4_beta:C
5assignvariableop_28_batch_normalization_4_moving_mean:G
9assignvariableop_29_batch_normalization_4_moving_variance:5
"assignvariableop_30_dense_7_kernel:	/
 assignvariableop_31_dense_7_bias:	6
"assignvariableop_32_dense_8_kernel:
/
 assignvariableop_33_dense_8_bias:	5
"assignvariableop_34_dense_6_kernel:	.
 assignvariableop_35_dense_6_bias:'
assignvariableop_36_iteration:	 +
!assignvariableop_37_learning_rate: D
*assignvariableop_38_adam_m_conv2d_6_kernel:D
*assignvariableop_39_adam_v_conv2d_6_kernel:6
(assignvariableop_40_adam_m_conv2d_6_bias:6
(assignvariableop_41_adam_v_conv2d_6_bias:B
4assignvariableop_42_adam_m_batch_normalization_gamma:B
4assignvariableop_43_adam_v_batch_normalization_gamma:A
3assignvariableop_44_adam_m_batch_normalization_beta:A
3assignvariableop_45_adam_v_batch_normalization_beta:D
*assignvariableop_46_adam_m_conv2d_7_kernel:D
*assignvariableop_47_adam_v_conv2d_7_kernel:6
(assignvariableop_48_adam_m_conv2d_7_bias:6
(assignvariableop_49_adam_v_conv2d_7_bias:D
6assignvariableop_50_adam_m_batch_normalization_1_gamma:D
6assignvariableop_51_adam_v_batch_normalization_1_gamma:C
5assignvariableop_52_adam_m_batch_normalization_1_beta:C
5assignvariableop_53_adam_v_batch_normalization_1_beta:D
*assignvariableop_54_adam_m_conv2d_8_kernel: D
*assignvariableop_55_adam_v_conv2d_8_kernel: 6
(assignvariableop_56_adam_m_conv2d_8_bias: 6
(assignvariableop_57_adam_v_conv2d_8_bias: D
6assignvariableop_58_adam_m_batch_normalization_2_gamma: D
6assignvariableop_59_adam_v_batch_normalization_2_gamma: C
5assignvariableop_60_adam_m_batch_normalization_2_beta: C
5assignvariableop_61_adam_v_batch_normalization_2_beta: D
*assignvariableop_62_adam_m_conv2d_9_kernel:  D
*assignvariableop_63_adam_v_conv2d_9_kernel:  6
(assignvariableop_64_adam_m_conv2d_9_bias: 6
(assignvariableop_65_adam_v_conv2d_9_bias: D
6assignvariableop_66_adam_m_batch_normalization_3_gamma: D
6assignvariableop_67_adam_v_batch_normalization_3_gamma: C
5assignvariableop_68_adam_m_batch_normalization_3_beta: C
5assignvariableop_69_adam_v_batch_normalization_3_beta: E
+assignvariableop_70_adam_m_conv2d_10_kernel: E
+assignvariableop_71_adam_v_conv2d_10_kernel: 7
)assignvariableop_72_adam_m_conv2d_10_bias:7
)assignvariableop_73_adam_v_conv2d_10_bias:D
6assignvariableop_74_adam_m_batch_normalization_4_gamma:D
6assignvariableop_75_adam_v_batch_normalization_4_gamma:C
5assignvariableop_76_adam_m_batch_normalization_4_beta:C
5assignvariableop_77_adam_v_batch_normalization_4_beta:<
)assignvariableop_78_adam_m_dense_7_kernel:	<
)assignvariableop_79_adam_v_dense_7_kernel:	6
'assignvariableop_80_adam_m_dense_7_bias:	6
'assignvariableop_81_adam_v_dense_7_bias:	=
)assignvariableop_82_adam_m_dense_8_kernel:
=
)assignvariableop_83_adam_v_dense_8_kernel:
6
'assignvariableop_84_adam_m_dense_8_bias:	6
'assignvariableop_85_adam_v_dense_8_bias:	<
)assignvariableop_86_adam_m_dense_6_kernel:	<
)assignvariableop_87_adam_v_dense_6_kernel:	5
'assignvariableop_88_adam_m_dense_6_bias:5
'assignvariableop_89_adam_v_dense_6_bias:%
assignvariableop_90_total_1: %
assignvariableop_91_count_1: #
assignvariableop_92_total: #
assignvariableop_93_count: 
identity_95ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93р(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*(
valueќ'Bљ'_B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*г
valueЩBЦ_B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ќ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesџ
ќ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*m
dtypesc
a2_	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_8_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_8_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_9_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_9_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_10_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_10_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_7_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_7_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_8_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_8_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_6_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_6_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_36AssignVariableOpassignvariableop_36_iterationIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_37AssignVariableOp!assignvariableop_37_learning_rateIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_conv2d_6_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_conv2d_6_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_conv2d_6_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_conv2d_6_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_m_batch_normalization_gammaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_v_batch_normalization_gammaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_m_batch_normalization_betaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adam_v_batch_normalization_betaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_m_conv2d_7_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_v_conv2d_7_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_m_conv2d_7_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_v_conv2d_7_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_m_batch_normalization_1_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_v_batch_normalization_1_gammaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_m_batch_normalization_1_betaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_v_batch_normalization_1_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_m_conv2d_8_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_v_conv2d_8_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_m_conv2d_8_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_v_conv2d_8_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_m_batch_normalization_2_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_v_batch_normalization_2_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_60AssignVariableOp5assignvariableop_60_adam_m_batch_normalization_2_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_v_batch_normalization_2_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_m_conv2d_9_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_v_conv2d_9_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_m_conv2d_9_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_v_conv2d_9_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_m_batch_normalization_3_gammaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_v_batch_normalization_3_gammaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_m_batch_normalization_3_betaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_v_batch_normalization_3_betaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_m_conv2d_10_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_v_conv2d_10_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_m_conv2d_10_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_v_conv2d_10_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_m_batch_normalization_4_gammaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_v_batch_normalization_4_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_76AssignVariableOp5assignvariableop_76_adam_m_batch_normalization_4_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_77AssignVariableOp5assignvariableop_77_adam_v_batch_normalization_4_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_m_dense_7_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_v_dense_7_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_m_dense_7_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_81AssignVariableOp'assignvariableop_81_adam_v_dense_7_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_m_dense_8_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adam_v_dense_8_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adam_m_dense_8_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_v_dense_8_biasIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_m_dense_6_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_87AssignVariableOp)assignvariableop_87_adam_v_dense_6_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_88AssignVariableOp'assignvariableop_88_adam_m_dense_6_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adam_v_dense_6_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_90AssignVariableOpassignvariableop_90_total_1Identity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_91AssignVariableOpassignvariableop_91_count_1Identity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_92AssignVariableOpassignvariableop_92_totalIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_93AssignVariableOpassignvariableop_93_countIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 у
Identity_94Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_95IdentityIdentity_94:output:0^NoOp_1*
T0*
_output_shapes
: Ќ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93*
_output_shapes
 "#
identity_95Identity_95:output:0*(
_construction_contextkEagerRuntime*г
_input_shapesС
О: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%^!

_user_specified_namecount:%]!

_user_specified_nametotal:'\#
!
_user_specified_name	count_1:'[#
!
_user_specified_name	total_1:3Z/
-
_user_specified_nameAdam/v/dense_6/bias:3Y/
-
_user_specified_nameAdam/m/dense_6/bias:5X1
/
_user_specified_nameAdam/v/dense_6/kernel:5W1
/
_user_specified_nameAdam/m/dense_6/kernel:3V/
-
_user_specified_nameAdam/v/dense_8/bias:3U/
-
_user_specified_nameAdam/m/dense_8/bias:5T1
/
_user_specified_nameAdam/v/dense_8/kernel:5S1
/
_user_specified_nameAdam/m/dense_8/kernel:3R/
-
_user_specified_nameAdam/v/dense_7/bias:3Q/
-
_user_specified_nameAdam/m/dense_7/bias:5P1
/
_user_specified_nameAdam/v/dense_7/kernel:5O1
/
_user_specified_nameAdam/m/dense_7/kernel:AN=
;
_user_specified_name#!Adam/v/batch_normalization_4/beta:AM=
;
_user_specified_name#!Adam/m/batch_normalization_4/beta:BL>
<
_user_specified_name$"Adam/v/batch_normalization_4/gamma:BK>
<
_user_specified_name$"Adam/m/batch_normalization_4/gamma:5J1
/
_user_specified_nameAdam/v/conv2d_10/bias:5I1
/
_user_specified_nameAdam/m/conv2d_10/bias:7H3
1
_user_specified_nameAdam/v/conv2d_10/kernel:7G3
1
_user_specified_nameAdam/m/conv2d_10/kernel:AF=
;
_user_specified_name#!Adam/v/batch_normalization_3/beta:AE=
;
_user_specified_name#!Adam/m/batch_normalization_3/beta:BD>
<
_user_specified_name$"Adam/v/batch_normalization_3/gamma:BC>
<
_user_specified_name$"Adam/m/batch_normalization_3/gamma:4B0
.
_user_specified_nameAdam/v/conv2d_9/bias:4A0
.
_user_specified_nameAdam/m/conv2d_9/bias:6@2
0
_user_specified_nameAdam/v/conv2d_9/kernel:6?2
0
_user_specified_nameAdam/m/conv2d_9/kernel:A>=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:A==
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:B<>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:B;>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:4:0
.
_user_specified_nameAdam/v/conv2d_8/bias:490
.
_user_specified_nameAdam/m/conv2d_8/bias:682
0
_user_specified_nameAdam/v/conv2d_8/kernel:672
0
_user_specified_nameAdam/m/conv2d_8/kernel:A6=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A5=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B4>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B3>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:420
.
_user_specified_nameAdam/v/conv2d_7/bias:410
.
_user_specified_nameAdam/m/conv2d_7/bias:602
0
_user_specified_nameAdam/v/conv2d_7/kernel:6/2
0
_user_specified_nameAdam/m/conv2d_7/kernel:?.;
9
_user_specified_name!Adam/v/batch_normalization/beta:?-;
9
_user_specified_name!Adam/m/batch_normalization/beta:@,<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@+<
:
_user_specified_name" Adam/m/batch_normalization/gamma:4*0
.
_user_specified_nameAdam/v/conv2d_6/bias:4)0
.
_user_specified_nameAdam/m/conv2d_6/bias:6(2
0
_user_specified_nameAdam/v/conv2d_6/kernel:6'2
0
_user_specified_nameAdam/m/conv2d_6/kernel:-&)
'
_user_specified_namelearning_rate:)%%
#
_user_specified_name	iteration:,$(
&
_user_specified_namedense_6/bias:.#*
(
_user_specified_namedense_6/kernel:,"(
&
_user_specified_namedense_8/bias:.!*
(
_user_specified_namedense_8/kernel:, (
&
_user_specified_namedense_7/bias:.*
(
_user_specified_namedense_7/kernel:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:A=
;
_user_specified_name#!batch_normalization_4/moving_mean::6
4
_user_specified_namebatch_normalization_4/beta:;7
5
_user_specified_namebatch_normalization_4/gamma:.*
(
_user_specified_nameconv2d_10/bias:0,
*
_user_specified_nameconv2d_10/kernel:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:A=
;
_user_specified_name#!batch_normalization_3/moving_mean::6
4
_user_specified_namebatch_normalization_3/beta:;7
5
_user_specified_namebatch_normalization_3/gamma:-)
'
_user_specified_nameconv2d_9/bias:/+
)
_user_specified_nameconv2d_9/kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:-)
'
_user_specified_nameconv2d_8/bias:/+
)
_user_specified_nameconv2d_8/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:-)
'
_user_specified_nameconv2d_7/bias:/+
)
_user_specified_nameconv2d_7/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:-)
'
_user_specified_nameconv2d_6/bias:/+
)
_user_specified_nameconv2d_6/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

e
G__inference_activation_1_layer_call_and_return_conditional_losses_49516

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..[
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..С
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49507*L
_output_shapes:
8:џџџџџџџџџ..:џџџџџџџџџ..: d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ.."!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ..:W S
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs


а
5__inference_batch_normalization_3_layer_call_fn_49685

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48367
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49681:%!

_user_specified_name49679:%!

_user_specified_name49677:%!

_user_specified_name49675:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_49931

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


Ю
3__inference_batch_normalization_layer_call_fn_49331

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_48151
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name49327:%!

_user_specified_name49325:%!

_user_specified_name49323:%!

_user_specified_name49321:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_6_layer_call_fn_49403

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_48200
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л

"__inference_internal_grad_fn_50742
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1l
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ..U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ..J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ..b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ..\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ..E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ..:џџџџџџџџџ..: : :џџџџџџџџџ..:XT
/
_output_shapes
:џџџџџџџџџ..
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ..
(
_user_specified_nameresult_grads_0
ЕЪ
Y
__inference__traced_save_51569
file_prefix@
&read_disablecopyonread_conv2d_6_kernel:4
&read_1_disablecopyonread_conv2d_6_bias:@
2read_2_disablecopyonread_batch_normalization_gamma:?
1read_3_disablecopyonread_batch_normalization_beta:F
8read_4_disablecopyonread_batch_normalization_moving_mean:J
<read_5_disablecopyonread_batch_normalization_moving_variance:B
(read_6_disablecopyonread_conv2d_7_kernel:4
&read_7_disablecopyonread_conv2d_7_bias:B
4read_8_disablecopyonread_batch_normalization_1_gamma:A
3read_9_disablecopyonread_batch_normalization_1_beta:I
;read_10_disablecopyonread_batch_normalization_1_moving_mean:M
?read_11_disablecopyonread_batch_normalization_1_moving_variance:C
)read_12_disablecopyonread_conv2d_8_kernel: 5
'read_13_disablecopyonread_conv2d_8_bias: C
5read_14_disablecopyonread_batch_normalization_2_gamma: B
4read_15_disablecopyonread_batch_normalization_2_beta: I
;read_16_disablecopyonread_batch_normalization_2_moving_mean: M
?read_17_disablecopyonread_batch_normalization_2_moving_variance: C
)read_18_disablecopyonread_conv2d_9_kernel:  5
'read_19_disablecopyonread_conv2d_9_bias: C
5read_20_disablecopyonread_batch_normalization_3_gamma: B
4read_21_disablecopyonread_batch_normalization_3_beta: I
;read_22_disablecopyonread_batch_normalization_3_moving_mean: M
?read_23_disablecopyonread_batch_normalization_3_moving_variance: D
*read_24_disablecopyonread_conv2d_10_kernel: 6
(read_25_disablecopyonread_conv2d_10_bias:C
5read_26_disablecopyonread_batch_normalization_4_gamma:B
4read_27_disablecopyonread_batch_normalization_4_beta:I
;read_28_disablecopyonread_batch_normalization_4_moving_mean:M
?read_29_disablecopyonread_batch_normalization_4_moving_variance:;
(read_30_disablecopyonread_dense_7_kernel:	5
&read_31_disablecopyonread_dense_7_bias:	<
(read_32_disablecopyonread_dense_8_kernel:
5
&read_33_disablecopyonread_dense_8_bias:	;
(read_34_disablecopyonread_dense_6_kernel:	4
&read_35_disablecopyonread_dense_6_bias:-
#read_36_disablecopyonread_iteration:	 1
'read_37_disablecopyonread_learning_rate: J
0read_38_disablecopyonread_adam_m_conv2d_6_kernel:J
0read_39_disablecopyonread_adam_v_conv2d_6_kernel:<
.read_40_disablecopyonread_adam_m_conv2d_6_bias:<
.read_41_disablecopyonread_adam_v_conv2d_6_bias:H
:read_42_disablecopyonread_adam_m_batch_normalization_gamma:H
:read_43_disablecopyonread_adam_v_batch_normalization_gamma:G
9read_44_disablecopyonread_adam_m_batch_normalization_beta:G
9read_45_disablecopyonread_adam_v_batch_normalization_beta:J
0read_46_disablecopyonread_adam_m_conv2d_7_kernel:J
0read_47_disablecopyonread_adam_v_conv2d_7_kernel:<
.read_48_disablecopyonread_adam_m_conv2d_7_bias:<
.read_49_disablecopyonread_adam_v_conv2d_7_bias:J
<read_50_disablecopyonread_adam_m_batch_normalization_1_gamma:J
<read_51_disablecopyonread_adam_v_batch_normalization_1_gamma:I
;read_52_disablecopyonread_adam_m_batch_normalization_1_beta:I
;read_53_disablecopyonread_adam_v_batch_normalization_1_beta:J
0read_54_disablecopyonread_adam_m_conv2d_8_kernel: J
0read_55_disablecopyonread_adam_v_conv2d_8_kernel: <
.read_56_disablecopyonread_adam_m_conv2d_8_bias: <
.read_57_disablecopyonread_adam_v_conv2d_8_bias: J
<read_58_disablecopyonread_adam_m_batch_normalization_2_gamma: J
<read_59_disablecopyonread_adam_v_batch_normalization_2_gamma: I
;read_60_disablecopyonread_adam_m_batch_normalization_2_beta: I
;read_61_disablecopyonread_adam_v_batch_normalization_2_beta: J
0read_62_disablecopyonread_adam_m_conv2d_9_kernel:  J
0read_63_disablecopyonread_adam_v_conv2d_9_kernel:  <
.read_64_disablecopyonread_adam_m_conv2d_9_bias: <
.read_65_disablecopyonread_adam_v_conv2d_9_bias: J
<read_66_disablecopyonread_adam_m_batch_normalization_3_gamma: J
<read_67_disablecopyonread_adam_v_batch_normalization_3_gamma: I
;read_68_disablecopyonread_adam_m_batch_normalization_3_beta: I
;read_69_disablecopyonread_adam_v_batch_normalization_3_beta: K
1read_70_disablecopyonread_adam_m_conv2d_10_kernel: K
1read_71_disablecopyonread_adam_v_conv2d_10_kernel: =
/read_72_disablecopyonread_adam_m_conv2d_10_bias:=
/read_73_disablecopyonread_adam_v_conv2d_10_bias:J
<read_74_disablecopyonread_adam_m_batch_normalization_4_gamma:J
<read_75_disablecopyonread_adam_v_batch_normalization_4_gamma:I
;read_76_disablecopyonread_adam_m_batch_normalization_4_beta:I
;read_77_disablecopyonread_adam_v_batch_normalization_4_beta:B
/read_78_disablecopyonread_adam_m_dense_7_kernel:	B
/read_79_disablecopyonread_adam_v_dense_7_kernel:	<
-read_80_disablecopyonread_adam_m_dense_7_bias:	<
-read_81_disablecopyonread_adam_v_dense_7_bias:	C
/read_82_disablecopyonread_adam_m_dense_8_kernel:
C
/read_83_disablecopyonread_adam_v_dense_8_kernel:
<
-read_84_disablecopyonread_adam_m_dense_8_bias:	<
-read_85_disablecopyonread_adam_v_dense_8_bias:	B
/read_86_disablecopyonread_adam_m_dense_6_kernel:	B
/read_87_disablecopyonread_adam_v_dense_6_kernel:	;
-read_88_disablecopyonread_adam_m_dense_6_bias:;
-read_89_disablecopyonread_adam_v_dense_6_bias:+
!read_90_disablecopyonread_total_1: +
!read_91_disablecopyonread_count_1: )
read_92_disablecopyonread_total: )
read_93_disablecopyonread_count: 
savev2_const
identity_189ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_52/DisableCopyOnReadЂRead_52/ReadVariableOpЂRead_53/DisableCopyOnReadЂRead_53/ReadVariableOpЂRead_54/DisableCopyOnReadЂRead_54/ReadVariableOpЂRead_55/DisableCopyOnReadЂRead_55/ReadVariableOpЂRead_56/DisableCopyOnReadЂRead_56/ReadVariableOpЂRead_57/DisableCopyOnReadЂRead_57/ReadVariableOpЂRead_58/DisableCopyOnReadЂRead_58/ReadVariableOpЂRead_59/DisableCopyOnReadЂRead_59/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_60/DisableCopyOnReadЂRead_60/ReadVariableOpЂRead_61/DisableCopyOnReadЂRead_61/ReadVariableOpЂRead_62/DisableCopyOnReadЂRead_62/ReadVariableOpЂRead_63/DisableCopyOnReadЂRead_63/ReadVariableOpЂRead_64/DisableCopyOnReadЂRead_64/ReadVariableOpЂRead_65/DisableCopyOnReadЂRead_65/ReadVariableOpЂRead_66/DisableCopyOnReadЂRead_66/ReadVariableOpЂRead_67/DisableCopyOnReadЂRead_67/ReadVariableOpЂRead_68/DisableCopyOnReadЂRead_68/ReadVariableOpЂRead_69/DisableCopyOnReadЂRead_69/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_70/DisableCopyOnReadЂRead_70/ReadVariableOpЂRead_71/DisableCopyOnReadЂRead_71/ReadVariableOpЂRead_72/DisableCopyOnReadЂRead_72/ReadVariableOpЂRead_73/DisableCopyOnReadЂRead_73/ReadVariableOpЂRead_74/DisableCopyOnReadЂRead_74/ReadVariableOpЂRead_75/DisableCopyOnReadЂRead_75/ReadVariableOpЂRead_76/DisableCopyOnReadЂRead_76/ReadVariableOpЂRead_77/DisableCopyOnReadЂRead_77/ReadVariableOpЂRead_78/DisableCopyOnReadЂRead_78/ReadVariableOpЂRead_79/DisableCopyOnReadЂRead_79/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_80/DisableCopyOnReadЂRead_80/ReadVariableOpЂRead_81/DisableCopyOnReadЂRead_81/ReadVariableOpЂRead_82/DisableCopyOnReadЂRead_82/ReadVariableOpЂRead_83/DisableCopyOnReadЂRead_83/ReadVariableOpЂRead_84/DisableCopyOnReadЂRead_84/ReadVariableOpЂRead_85/DisableCopyOnReadЂRead_85/ReadVariableOpЂRead_86/DisableCopyOnReadЂRead_86/ReadVariableOpЂRead_87/DisableCopyOnReadЂRead_87/ReadVariableOpЂRead_88/DisableCopyOnReadЂRead_88/ReadVariableOpЂRead_89/DisableCopyOnReadЂRead_89/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpЂRead_90/DisableCopyOnReadЂRead_90/ReadVariableOpЂRead_91/DisableCopyOnReadЂRead_91/ReadVariableOpЂRead_92/DisableCopyOnReadЂRead_92/ReadVariableOpЂRead_93/DisableCopyOnReadЂRead_93/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 Њ
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_6_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_6_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_2/DisableCopyOnReadDisableCopyOnRead2read_2_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Ў
Read_2/ReadVariableOpReadVariableOp2read_2_disablecopyonread_batch_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 ­
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_batch_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_4/DisableCopyOnReadDisableCopyOnRead8read_4_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 Д
Read_4/ReadVariableOpReadVariableOp8read_4_disablecopyonread_batch_normalization_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_5/DisableCopyOnReadDisableCopyOnRead<read_5_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 И
Read_5/ReadVariableOpReadVariableOp<read_5_disablecopyonread_batch_normalization_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 А
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_7_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_7_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 А
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 Џ
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 Г
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv2d_8_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv2d_8_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv2d_8_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 Г
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_2_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 В
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_2_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_2_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_2_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 Г
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_conv2d_9_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
:  |
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_conv2d_9_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_20/DisableCopyOnReadDisableCopyOnRead5read_20_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 Г
Read_20/ReadVariableOpReadVariableOp5read_20_disablecopyonread_batch_normalization_3_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 В
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_batch_normalization_3_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_22/DisableCopyOnReadDisableCopyOnRead;read_22_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_22/ReadVariableOpReadVariableOp;read_22_disablecopyonread_batch_normalization_3_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_23/DisableCopyOnReadDisableCopyOnRead?read_23_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_23/ReadVariableOpReadVariableOp?read_23_disablecopyonread_batch_normalization_3_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_24/DisableCopyOnReadDisableCopyOnRead*read_24_disablecopyonread_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 Д
Read_24/ReadVariableOpReadVariableOp*read_24_disablecopyonread_conv2d_10_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
: }
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_conv2d_10_bias"/device:CPU:0*
_output_shapes
 І
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_conv2d_10_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_26/DisableCopyOnReadDisableCopyOnRead5read_26_disablecopyonread_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 Г
Read_26/ReadVariableOpReadVariableOp5read_26_disablecopyonread_batch_normalization_4_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_27/DisableCopyOnReadDisableCopyOnRead4read_27_disablecopyonread_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 В
Read_27/ReadVariableOpReadVariableOp4read_27_disablecopyonread_batch_normalization_4_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_28/DisableCopyOnReadDisableCopyOnRead;read_28_disablecopyonread_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_28/ReadVariableOpReadVariableOp;read_28_disablecopyonread_batch_normalization_4_moving_mean^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_29/DisableCopyOnReadDisableCopyOnRead?read_29_disablecopyonread_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_29/ReadVariableOpReadVariableOp?read_29_disablecopyonread_batch_normalization_4_moving_variance^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_30/DisableCopyOnReadDisableCopyOnRead(read_30_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_30/ReadVariableOpReadVariableOp(read_30_disablecopyonread_dense_7_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	{
Read_31/DisableCopyOnReadDisableCopyOnRead&read_31_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_31/ReadVariableOpReadVariableOp&read_31_disablecopyonread_dense_7_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:}
Read_32/DisableCopyOnReadDisableCopyOnRead(read_32_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_32/ReadVariableOpReadVariableOp(read_32_disablecopyonread_dense_8_kernel^Read_32/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0* 
_output_shapes
:
{
Read_33/DisableCopyOnReadDisableCopyOnRead&read_33_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_33/ReadVariableOpReadVariableOp&read_33_disablecopyonread_dense_8_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:}
Read_34/DisableCopyOnReadDisableCopyOnRead(read_34_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_34/ReadVariableOpReadVariableOp(read_34_disablecopyonread_dense_6_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	{
Read_35/DisableCopyOnReadDisableCopyOnRead&read_35_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 Є
Read_35/ReadVariableOpReadVariableOp&read_35_disablecopyonread_dense_6_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_36/DisableCopyOnReadDisableCopyOnRead#read_36_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_36/ReadVariableOpReadVariableOp#read_36_disablecopyonread_iteration^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_37/DisableCopyOnReadDisableCopyOnRead'read_37_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_37/ReadVariableOpReadVariableOp'read_37_disablecopyonread_learning_rate^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 К
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_conv2d_6_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 К
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_conv2d_6_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_conv2d_6_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_conv2d_6_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_conv2d_6_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_42/DisableCopyOnReadDisableCopyOnRead:read_42_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 И
Read_42/ReadVariableOpReadVariableOp:read_42_disablecopyonread_adam_m_batch_normalization_gamma^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_43/DisableCopyOnReadDisableCopyOnRead:read_43_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 И
Read_43/ReadVariableOpReadVariableOp:read_43_disablecopyonread_adam_v_batch_normalization_gamma^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_44/DisableCopyOnReadDisableCopyOnRead9read_44_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 З
Read_44/ReadVariableOpReadVariableOp9read_44_disablecopyonread_adam_m_batch_normalization_beta^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_45/DisableCopyOnReadDisableCopyOnRead9read_45_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 З
Read_45/ReadVariableOpReadVariableOp9read_45_disablecopyonread_adam_v_batch_normalization_beta^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_m_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 К
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_m_conv2d_7_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_v_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 К
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_v_conv2d_7_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_adam_m_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_adam_m_conv2d_7_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_49/DisableCopyOnReadDisableCopyOnRead.read_49_disablecopyonread_adam_v_conv2d_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_49/ReadVariableOpReadVariableOp.read_49_disablecopyonread_adam_v_conv2d_7_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_50/DisableCopyOnReadDisableCopyOnRead<read_50_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 К
Read_50/ReadVariableOpReadVariableOp<read_50_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_51/DisableCopyOnReadDisableCopyOnRead<read_51_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 К
Read_51/ReadVariableOpReadVariableOp<read_51_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_52/DisableCopyOnReadDisableCopyOnRead;read_52_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 Й
Read_52/ReadVariableOpReadVariableOp;read_52_disablecopyonread_adam_m_batch_normalization_1_beta^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_53/DisableCopyOnReadDisableCopyOnRead;read_53_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 Й
Read_53/ReadVariableOpReadVariableOp;read_53_disablecopyonread_adam_v_batch_normalization_1_beta^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_54/DisableCopyOnReadDisableCopyOnRead0read_54_disablecopyonread_adam_m_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 К
Read_54/ReadVariableOpReadVariableOp0read_54_disablecopyonread_adam_m_conv2d_8_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_55/DisableCopyOnReadDisableCopyOnRead0read_55_disablecopyonread_adam_v_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 К
Read_55/ReadVariableOpReadVariableOp0read_55_disablecopyonread_adam_v_conv2d_8_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_56/DisableCopyOnReadDisableCopyOnRead.read_56_disablecopyonread_adam_m_conv2d_8_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_56/ReadVariableOpReadVariableOp.read_56_disablecopyonread_adam_m_conv2d_8_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_57/DisableCopyOnReadDisableCopyOnRead.read_57_disablecopyonread_adam_v_conv2d_8_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_57/ReadVariableOpReadVariableOp.read_57_disablecopyonread_adam_v_conv2d_8_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_58/DisableCopyOnReadDisableCopyOnRead<read_58_disablecopyonread_adam_m_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 К
Read_58/ReadVariableOpReadVariableOp<read_58_disablecopyonread_adam_m_batch_normalization_2_gamma^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_59/DisableCopyOnReadDisableCopyOnRead<read_59_disablecopyonread_adam_v_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 К
Read_59/ReadVariableOpReadVariableOp<read_59_disablecopyonread_adam_v_batch_normalization_2_gamma^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_60/DisableCopyOnReadDisableCopyOnRead;read_60_disablecopyonread_adam_m_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 Й
Read_60/ReadVariableOpReadVariableOp;read_60_disablecopyonread_adam_m_batch_normalization_2_beta^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_61/DisableCopyOnReadDisableCopyOnRead;read_61_disablecopyonread_adam_v_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 Й
Read_61/ReadVariableOpReadVariableOp;read_61_disablecopyonread_adam_v_batch_normalization_2_beta^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_62/DisableCopyOnReadDisableCopyOnRead0read_62_disablecopyonread_adam_m_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 К
Read_62/ReadVariableOpReadVariableOp0read_62_disablecopyonread_adam_m_conv2d_9_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*&
_output_shapes
:  
Read_63/DisableCopyOnReadDisableCopyOnRead0read_63_disablecopyonread_adam_v_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 К
Read_63/ReadVariableOpReadVariableOp0read_63_disablecopyonread_adam_v_conv2d_9_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*&
_output_shapes
:  
Read_64/DisableCopyOnReadDisableCopyOnRead.read_64_disablecopyonread_adam_m_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_64/ReadVariableOpReadVariableOp.read_64_disablecopyonread_adam_m_conv2d_9_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_65/DisableCopyOnReadDisableCopyOnRead.read_65_disablecopyonread_adam_v_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_65/ReadVariableOpReadVariableOp.read_65_disablecopyonread_adam_v_conv2d_9_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_66/DisableCopyOnReadDisableCopyOnRead<read_66_disablecopyonread_adam_m_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 К
Read_66/ReadVariableOpReadVariableOp<read_66_disablecopyonread_adam_m_batch_normalization_3_gamma^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_67/DisableCopyOnReadDisableCopyOnRead<read_67_disablecopyonread_adam_v_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 К
Read_67/ReadVariableOpReadVariableOp<read_67_disablecopyonread_adam_v_batch_normalization_3_gamma^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_68/DisableCopyOnReadDisableCopyOnRead;read_68_disablecopyonread_adam_m_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Й
Read_68/ReadVariableOpReadVariableOp;read_68_disablecopyonread_adam_m_batch_normalization_3_beta^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_69/DisableCopyOnReadDisableCopyOnRead;read_69_disablecopyonread_adam_v_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Й
Read_69/ReadVariableOpReadVariableOp;read_69_disablecopyonread_adam_v_batch_normalization_3_beta^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_70/DisableCopyOnReadDisableCopyOnRead1read_70_disablecopyonread_adam_m_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 Л
Read_70/ReadVariableOpReadVariableOp1read_70_disablecopyonread_adam_m_conv2d_10_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_71/DisableCopyOnReadDisableCopyOnRead1read_71_disablecopyonread_adam_v_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 Л
Read_71/ReadVariableOpReadVariableOp1read_71_disablecopyonread_adam_v_conv2d_10_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_72/DisableCopyOnReadDisableCopyOnRead/read_72_disablecopyonread_adam_m_conv2d_10_bias"/device:CPU:0*
_output_shapes
 ­
Read_72/ReadVariableOpReadVariableOp/read_72_disablecopyonread_adam_m_conv2d_10_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_73/DisableCopyOnReadDisableCopyOnRead/read_73_disablecopyonread_adam_v_conv2d_10_bias"/device:CPU:0*
_output_shapes
 ­
Read_73/ReadVariableOpReadVariableOp/read_73_disablecopyonread_adam_v_conv2d_10_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_74/DisableCopyOnReadDisableCopyOnRead<read_74_disablecopyonread_adam_m_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 К
Read_74/ReadVariableOpReadVariableOp<read_74_disablecopyonread_adam_m_batch_normalization_4_gamma^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_75/DisableCopyOnReadDisableCopyOnRead<read_75_disablecopyonread_adam_v_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 К
Read_75/ReadVariableOpReadVariableOp<read_75_disablecopyonread_adam_v_batch_normalization_4_gamma^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_76/DisableCopyOnReadDisableCopyOnRead;read_76_disablecopyonread_adam_m_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 Й
Read_76/ReadVariableOpReadVariableOp;read_76_disablecopyonread_adam_m_batch_normalization_4_beta^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_77/DisableCopyOnReadDisableCopyOnRead;read_77_disablecopyonread_adam_v_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 Й
Read_77/ReadVariableOpReadVariableOp;read_77_disablecopyonread_adam_v_batch_normalization_4_beta^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_78/DisableCopyOnReadDisableCopyOnRead/read_78_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 В
Read_78/ReadVariableOpReadVariableOp/read_78_disablecopyonread_adam_m_dense_7_kernel^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0q
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	h
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_79/DisableCopyOnReadDisableCopyOnRead/read_79_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 В
Read_79/ReadVariableOpReadVariableOp/read_79_disablecopyonread_adam_v_dense_7_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0q
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	h
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_80/DisableCopyOnReadDisableCopyOnRead-read_80_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_80/ReadVariableOpReadVariableOp-read_80_disablecopyonread_adam_m_dense_7_bias^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0m
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:d
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_81/DisableCopyOnReadDisableCopyOnRead-read_81_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_81/ReadVariableOpReadVariableOp-read_81_disablecopyonread_adam_v_dense_7_bias^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0m
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:d
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_82/DisableCopyOnReadDisableCopyOnRead/read_82_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 Г
Read_82/ReadVariableOpReadVariableOp/read_82_disablecopyonread_adam_m_dense_8_kernel^Read_82/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0r
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
i
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_83/DisableCopyOnReadDisableCopyOnRead/read_83_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 Г
Read_83/ReadVariableOpReadVariableOp/read_83_disablecopyonread_adam_v_dense_8_kernel^Read_83/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0r
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
i
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_84/DisableCopyOnReadDisableCopyOnRead-read_84_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_84/ReadVariableOpReadVariableOp-read_84_disablecopyonread_adam_m_dense_8_bias^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0m
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:d
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_85/DisableCopyOnReadDisableCopyOnRead-read_85_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_85/ReadVariableOpReadVariableOp-read_85_disablecopyonread_adam_v_dense_8_bias^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0m
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:d
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_86/DisableCopyOnReadDisableCopyOnRead/read_86_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 В
Read_86/ReadVariableOpReadVariableOp/read_86_disablecopyonread_adam_m_dense_6_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0q
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	h
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_87/DisableCopyOnReadDisableCopyOnRead/read_87_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 В
Read_87/ReadVariableOpReadVariableOp/read_87_disablecopyonread_adam_v_dense_6_kernel^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0q
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	h
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_88/DisableCopyOnReadDisableCopyOnRead-read_88_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_88/ReadVariableOpReadVariableOp-read_88_disablecopyonread_adam_m_dense_6_bias^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_89/DisableCopyOnReadDisableCopyOnRead-read_89_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_89/ReadVariableOpReadVariableOp-read_89_disablecopyonread_adam_v_dense_6_bias^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_90/DisableCopyOnReadDisableCopyOnRead!read_90_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_90/ReadVariableOpReadVariableOp!read_90_disablecopyonread_total_1^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_91/DisableCopyOnReadDisableCopyOnRead!read_91_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_91/ReadVariableOpReadVariableOp!read_91_disablecopyonread_count_1^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_92/DisableCopyOnReadDisableCopyOnReadread_92_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_92/ReadVariableOpReadVariableOpread_92_disablecopyonread_total^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_93/DisableCopyOnReadDisableCopyOnReadread_93_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_93/ReadVariableOpReadVariableOpread_93_disablecopyonread_count^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
: н(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*(
valueќ'Bљ'_B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:_*
dtype0*г
valueЩBЦ_B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *m
dtypesc
a2_	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_188Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_189IdentityIdentity_188:output:0^NoOp*
T0*
_output_shapes
: '
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp*
_output_shapes
 "%
identity_189Identity_189:output:0*(
_construction_contextkEagerRuntime*е
_input_shapesУ
Р: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp:=_9

_output_shapes
: 

_user_specified_nameConst:%^!

_user_specified_namecount:%]!

_user_specified_nametotal:'\#
!
_user_specified_name	count_1:'[#
!
_user_specified_name	total_1:3Z/
-
_user_specified_nameAdam/v/dense_6/bias:3Y/
-
_user_specified_nameAdam/m/dense_6/bias:5X1
/
_user_specified_nameAdam/v/dense_6/kernel:5W1
/
_user_specified_nameAdam/m/dense_6/kernel:3V/
-
_user_specified_nameAdam/v/dense_8/bias:3U/
-
_user_specified_nameAdam/m/dense_8/bias:5T1
/
_user_specified_nameAdam/v/dense_8/kernel:5S1
/
_user_specified_nameAdam/m/dense_8/kernel:3R/
-
_user_specified_nameAdam/v/dense_7/bias:3Q/
-
_user_specified_nameAdam/m/dense_7/bias:5P1
/
_user_specified_nameAdam/v/dense_7/kernel:5O1
/
_user_specified_nameAdam/m/dense_7/kernel:AN=
;
_user_specified_name#!Adam/v/batch_normalization_4/beta:AM=
;
_user_specified_name#!Adam/m/batch_normalization_4/beta:BL>
<
_user_specified_name$"Adam/v/batch_normalization_4/gamma:BK>
<
_user_specified_name$"Adam/m/batch_normalization_4/gamma:5J1
/
_user_specified_nameAdam/v/conv2d_10/bias:5I1
/
_user_specified_nameAdam/m/conv2d_10/bias:7H3
1
_user_specified_nameAdam/v/conv2d_10/kernel:7G3
1
_user_specified_nameAdam/m/conv2d_10/kernel:AF=
;
_user_specified_name#!Adam/v/batch_normalization_3/beta:AE=
;
_user_specified_name#!Adam/m/batch_normalization_3/beta:BD>
<
_user_specified_name$"Adam/v/batch_normalization_3/gamma:BC>
<
_user_specified_name$"Adam/m/batch_normalization_3/gamma:4B0
.
_user_specified_nameAdam/v/conv2d_9/bias:4A0
.
_user_specified_nameAdam/m/conv2d_9/bias:6@2
0
_user_specified_nameAdam/v/conv2d_9/kernel:6?2
0
_user_specified_nameAdam/m/conv2d_9/kernel:A>=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:A==
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:B<>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:B;>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:4:0
.
_user_specified_nameAdam/v/conv2d_8/bias:490
.
_user_specified_nameAdam/m/conv2d_8/bias:682
0
_user_specified_nameAdam/v/conv2d_8/kernel:672
0
_user_specified_nameAdam/m/conv2d_8/kernel:A6=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A5=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B4>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B3>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:420
.
_user_specified_nameAdam/v/conv2d_7/bias:410
.
_user_specified_nameAdam/m/conv2d_7/bias:602
0
_user_specified_nameAdam/v/conv2d_7/kernel:6/2
0
_user_specified_nameAdam/m/conv2d_7/kernel:?.;
9
_user_specified_name!Adam/v/batch_normalization/beta:?-;
9
_user_specified_name!Adam/m/batch_normalization/beta:@,<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@+<
:
_user_specified_name" Adam/m/batch_normalization/gamma:4*0
.
_user_specified_nameAdam/v/conv2d_6/bias:4)0
.
_user_specified_nameAdam/m/conv2d_6/bias:6(2
0
_user_specified_nameAdam/v/conv2d_6/kernel:6'2
0
_user_specified_nameAdam/m/conv2d_6/kernel:-&)
'
_user_specified_namelearning_rate:)%%
#
_user_specified_name	iteration:,$(
&
_user_specified_namedense_6/bias:.#*
(
_user_specified_namedense_6/kernel:,"(
&
_user_specified_namedense_8/bias:.!*
(
_user_specified_namedense_8/kernel:, (
&
_user_specified_namedense_7/bias:.*
(
_user_specified_namedense_7/kernel:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:A=
;
_user_specified_name#!batch_normalization_4/moving_mean::6
4
_user_specified_namebatch_normalization_4/beta:;7
5
_user_specified_namebatch_normalization_4/gamma:.*
(
_user_specified_nameconv2d_10/bias:0,
*
_user_specified_nameconv2d_10/kernel:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:A=
;
_user_specified_name#!batch_normalization_3/moving_mean::6
4
_user_specified_namebatch_normalization_3/beta:;7
5
_user_specified_namebatch_normalization_3/gamma:-)
'
_user_specified_nameconv2d_9/bias:/+
)
_user_specified_nameconv2d_9/kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:-)
'
_user_specified_nameconv2d_8/bias:/+
)
_user_specified_nameconv2d_8/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:-)
'
_user_specified_nameconv2d_7/bias:/+
)
_user_specified_nameconv2d_7/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:-)
'
_user_specified_nameconv2d_6/bias:/+
)
_user_specified_nameconv2d_6/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

П
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_48295

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50472
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ V
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :WS
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0

П
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48223

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж

"__inference_internal_grad_fn_50553
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1k
mulMulmul_beta
mul_inputs^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_1Mulmul_beta
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ V
SquareSquare
mul_inputs*
T0*/
_output_shapes
:џџџџџџџџџ b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: a
mul_7Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :WS
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
/
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0

П
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49480

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_49408

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ч
"__inference_internal_grad_fn_51147
result_grads_0
result_grads_1
result_grads_2!
mul_sequential_2_dense_7_beta$
 mul_sequential_2_dense_7_biasadd
identity

identity_1
mulMulmul_sequential_2_dense_7_beta mul_sequential_2_dense_7_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
mul_1Mulmul_sequential_2_dense_7_beta mul_sequential_2_dense_7_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџe
SquareSquare mul_sequential_2_dense_7_biasadd*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:џџџџџџџџџW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Z
mul_7Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:fb
(
_output_shapes
:џџџџџџџџџ
6
_user_specified_namesequential_2/dense_7/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namesequential_2/dense_7/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: |
&
 _has_manual_control_dependencies(
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
Я

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_48385

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_49870

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
G__inference_activation_3_layer_call_and_return_conditional_losses_49752

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
mulMulbeta:output:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ [
mul_1MulinputsSigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ С
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*+
_gradient_op_typeCustomGradient-49743*L
_output_shapes:
8:џџџџџџџџџ :џџџџџџџџџ : d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:
"__inference_internal_grad_fn_50229CustomGradient-49955:
"__inference_internal_grad_fn_50256CustomGradient-48778:
"__inference_internal_grad_fn_50283CustomGradient-49900:
"__inference_internal_grad_fn_50310CustomGradient-48741:
"__inference_internal_grad_fn_50337CustomGradient-49851:
"__inference_internal_grad_fn_50364CustomGradient-48713:
"__inference_internal_grad_fn_50391CustomGradient-49771:
"__inference_internal_grad_fn_50418CustomGradient-48686:
"__inference_internal_grad_fn_50445CustomGradient-49743:
"__inference_internal_grad_fn_50472CustomGradient-48666:
"__inference_internal_grad_fn_50499CustomGradient-49663:
"__inference_internal_grad_fn_50526CustomGradient-48639:
"__inference_internal_grad_fn_50553CustomGradient-49625:
"__inference_internal_grad_fn_50580CustomGradient-48618:
"__inference_internal_grad_fn_50607CustomGradient-49545:
"__inference_internal_grad_fn_50634CustomGradient-48591:
"__inference_internal_grad_fn_50661CustomGradient-49507:
"__inference_internal_grad_fn_50688CustomGradient-48570:
"__inference_internal_grad_fn_50715CustomGradient-49427:
"__inference_internal_grad_fn_50742CustomGradient-48543:
"__inference_internal_grad_fn_50769CustomGradient-49389:
"__inference_internal_grad_fn_50796CustomGradient-48522:
"__inference_internal_grad_fn_50823CustomGradient-49309:
"__inference_internal_grad_fn_50850CustomGradient-48495:
"__inference_internal_grad_fn_50877CustomGradient-47904:
"__inference_internal_grad_fn_50904CustomGradient-47927:
"__inference_internal_grad_fn_50931CustomGradient-47943:
"__inference_internal_grad_fn_50958CustomGradient-47966:
"__inference_internal_grad_fn_50985CustomGradient-47982:
"__inference_internal_grad_fn_51012CustomGradient-48005:
"__inference_internal_grad_fn_51039CustomGradient-48021:
"__inference_internal_grad_fn_51066CustomGradient-48044:
"__inference_internal_grad_fn_51093CustomGradient-48059:
"__inference_internal_grad_fn_51120CustomGradient-48082:
"__inference_internal_grad_fn_51147CustomGradient-48100:
"__inference_internal_grad_fn_51174CustomGradient-48116"ЇL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_default
C
input_38
serving_default_input_3:0џџџџџџџџџpp;
dense_60
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЕО

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer_with_weights-12
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures"
_tf_keras_sequential
н
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op"
_tf_keras_layer
ъ
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2axis
	3gamma
4beta
5moving_mean
6moving_variance"
_tf_keras_layer
Ѕ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
н
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op"
_tf_keras_layer
ъ
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance"
_tf_keras_layer
Ѕ
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
н
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias
 k_jit_compiled_convolution_op"
_tf_keras_layer
ъ
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance"
_tf_keras_layer
Ѕ
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
Ј
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses
Ѓkernel
	Єbias
!Ѕ_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses
	Ќaxis

­gamma
	Ўbeta
Џmoving_mean
Аmoving_variance"
_tf_keras_layer
Ћ
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
У
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щkernel
	Ъbias"
_tf_keras_layer
У
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
б_random_generator"
_tf_keras_layer
У
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
иkernel
	йbias"
_tf_keras_layer
У
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses
р_random_generator"
_tf_keras_layer
У
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
чkernel
	шbias"
_tf_keras_layer
Ш
)0
*1
32
43
54
65
I6
J7
S8
T9
U10
V11
i12
j13
s14
t15
u16
v17
18
19
20
21
22
23
Ѓ24
Є25
­26
Ў27
Џ28
А29
Щ30
Ъ31
и32
й33
ч34
ш35"
trackable_list_wrapper
є
)0
*1
32
43
I4
J5
S6
T7
i8
j9
s10
t11
12
13
14
15
Ѓ16
Є17
­18
Ў19
Щ20
Ъ21
и22
й23
ч24
ш25"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
юtrace_0
яtrace_12
,__inference_sequential_2_layer_call_fn_49011
,__inference_sequential_2_layer_call_fn_49088Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zюtrace_0zяtrace_1

№trace_0
ёtrace_12Ъ
G__inference_sequential_2_layer_call_and_return_conditional_losses_48823
G__inference_sequential_2_layer_call_and_return_conditional_losses_48934Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z№trace_0zёtrace_1
ЫBШ
 __inference__wrapped_model_48133input_3"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ
ђ
_variables
ѓ_iterations
є_learning_rate
ѕ_index_dict
і
_momentums
ї_velocities
ј_update_step_xla"
experimentalOptimizer
-
љserving_default"
signature_map
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ф
џtrace_02Х
(__inference_conv2d_6_layer_call_fn_49299
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0
џ
trace_02р
C__inference_conv2d_6_layer_call_and_return_conditional_losses_49318
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):'2conv2d_6/kernel
:2conv2d_6/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
н
trace_0
trace_12Ђ
3__inference_batch_normalization_layer_call_fn_49331
3__inference_batch_normalization_layer_call_fn_49344Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12и
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49362
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49380Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_activation_layer_call_fn_49385
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02т
E__inference_activation_layer_call_and_return_conditional_losses_49398
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
/__inference_max_pooling2d_6_layer_call_fn_49403
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ч
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_49408
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
ф
trace_02Х
(__inference_conv2d_7_layer_call_fn_49417
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
џ
trace_02р
C__inference_conv2d_7_layer_call_and_return_conditional_losses_49436
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
):'2conv2d_7/kernel
:2conv2d_7/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
S0
T1
U2
V3"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
с
Єtrace_0
Ѕtrace_12І
5__inference_batch_normalization_1_layer_call_fn_49449
5__inference_batch_normalization_1_layer_call_fn_49462Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЄtrace_0zЅtrace_1

Іtrace_0
Їtrace_12м
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49480
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49498Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0zЇtrace_1
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
ш
­trace_02Щ
,__inference_activation_1_layer_call_fn_49503
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0

Ўtrace_02ф
G__inference_activation_1_layer_call_and_return_conditional_losses_49516
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
ы
Дtrace_02Ь
/__inference_max_pooling2d_7_layer_call_fn_49521
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0

Еtrace_02ч
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_49526
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
ф
Лtrace_02Х
(__inference_conv2d_8_layer_call_fn_49535
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0
џ
Мtrace_02р
C__inference_conv2d_8_layer_call_and_return_conditional_losses_49554
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0
):' 2conv2d_8/kernel
: 2conv2d_8/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
s0
t1
u2
v3"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
с
Тtrace_0
Уtrace_12І
5__inference_batch_normalization_2_layer_call_fn_49567
5__inference_batch_normalization_2_layer_call_fn_49580Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0zУtrace_1

Фtrace_0
Хtrace_12м
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49598
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49616Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0zХtrace_1
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
ш
Ыtrace_02Щ
,__inference_activation_2_layer_call_fn_49621
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЫtrace_0

Ьtrace_02ф
G__inference_activation_2_layer_call_and_return_conditional_losses_49634
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ы
вtrace_02Ь
/__inference_max_pooling2d_8_layer_call_fn_49639
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zвtrace_0

гtrace_02ч
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_49644
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ф
йtrace_02Х
(__inference_conv2d_9_layer_call_fn_49653
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0
џ
кtrace_02р
C__inference_conv2d_9_layer_call_and_return_conditional_losses_49672
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0
):'  2conv2d_9/kernel
: 2conv2d_9/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
с
рtrace_0
сtrace_12І
5__inference_batch_normalization_3_layer_call_fn_49685
5__inference_batch_normalization_3_layer_call_fn_49698Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zрtrace_0zсtrace_1

тtrace_0
уtrace_12м
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49716
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49734Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zтtrace_0zуtrace_1
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ш
щtrace_02Щ
,__inference_activation_3_layer_call_fn_49739
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zщtrace_0

ъtrace_02ф
G__inference_activation_3_layer_call_and_return_conditional_losses_49752
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0
0
Ѓ0
Є1"
trackable_list_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
х
№trace_02Ц
)__inference_conv2d_10_layer_call_fn_49761
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z№trace_0

ёtrace_02с
D__inference_conv2d_10_layer_call_and_return_conditional_losses_49780
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0
*:( 2conv2d_10/kernel
:2conv2d_10/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
­0
Ў1
Џ2
А3"
trackable_list_wrapper
0
­0
Ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
с
їtrace_0
јtrace_12І
5__inference_batch_normalization_4_layer_call_fn_49793
5__inference_batch_normalization_4_layer_call_fn_49806Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0zјtrace_1

љtrace_0
њtrace_12м
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49824
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49842Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zљtrace_0zњtrace_1
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
ш
trace_02Щ
,__inference_activation_4_layer_call_fn_49847
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ф
G__inference_activation_4_layer_call_and_return_conditional_losses_49860
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
/__inference_max_pooling2d_9_layer_call_fn_49865
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ч
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_49870
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_flatten_2_layer_call_fn_49875
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02с
D__inference_flatten_2_layer_call_and_return_conditional_losses_49881
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
Щ0
Ъ1"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
у
trace_02Ф
'__inference_dense_7_layer_call_fn_49890
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ў
trace_02п
B__inference_dense_7_layer_call_and_return_conditional_losses_49909
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
!:	2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
Н
trace_0
trace_12
)__inference_dropout_4_layer_call_fn_49914
)__inference_dropout_4_layer_call_fn_49919Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ѓ
trace_0
trace_12И
D__inference_dropout_4_layer_call_and_return_conditional_losses_49931
D__inference_dropout_4_layer_call_and_return_conditional_losses_49936Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
0
и0
й1"
trackable_list_wrapper
0
и0
й1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
у
Ѕtrace_02Ф
'__inference_dense_8_layer_call_fn_49945
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0
ў
Іtrace_02п
B__inference_dense_8_layer_call_and_return_conditional_losses_49964
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0
": 
2dense_8/kernel
:2dense_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
Н
Ќtrace_0
­trace_12
)__inference_dropout_5_layer_call_fn_49969
)__inference_dropout_5_layer_call_fn_49974Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0z­trace_1
ѓ
Ўtrace_0
Џtrace_12И
D__inference_dropout_5_layer_call_and_return_conditional_losses_49986
D__inference_dropout_5_layer_call_and_return_conditional_losses_49991Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0zЏtrace_1
"
_generic_user_object
0
ч0
ш1"
trackable_list_wrapper
0
ч0
ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
у
Еtrace_02Ф
'__inference_dense_6_layer_call_fn_50000
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
ў
Жtrace_02п
B__inference_dense_6_layer_call_and_return_conditional_losses_50011
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0
!:	2dense_6/kernel
:2dense_6/bias
j
50
61
U2
V3
u4
v5
6
7
Џ8
А9"
trackable_list_wrapper
о
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
,__inference_sequential_2_layer_call_fn_49011input_3"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
,__inference_sequential_2_layer_call_fn_49088input_3"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_2_layer_call_and_return_conditional_losses_48823input_3"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_2_layer_call_and_return_conditional_losses_48934input_3"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ
ѓ0
Й1
К2
Л3
М4
Н5
О6
П7
Р8
С9
Т10
У11
Ф12
Х13
Ц14
Ч15
Ш16
Щ17
Ъ18
Ы19
Ь20
Э21
Ю22
Я23
а24
б25
в26
г27
д28
е29
ж30
з31
и32
й33
к34
л35
м36
н37
о38
п39
р40
с41
т42
у43
ф44
х45
ц46
ч47
ш48
щ49
ъ50
ы51
ь52"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper

Й0
Л1
Н2
П3
С4
У5
Х6
Ч7
Щ8
Ы9
Э10
Я11
б12
г13
е14
з15
й16
л17
н18
п19
с20
у21
х22
ч23
щ24
ы25"
trackable_list_wrapper

К0
М1
О2
Р3
Т4
Ф5
Ц6
Ш7
Ъ8
Ь9
Ю10
а11
в12
д13
ж14
и15
к16
м17
о18
р19
т20
ф21
ц22
ш23
ъ24
ь25"
trackable_list_wrapper
Е2ВЏ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
ЯBЬ
#__inference_signature_wrapper_49290input_3"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs
	jinput_3
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv2d_6_layer_call_fn_49299inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv2d_6_layer_call_and_return_conditional_losses_49318inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
3__inference_batch_normalization_layer_call_fn_49331inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
3__inference_batch_normalization_layer_call_fn_49344inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49362inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49380inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_activation_layer_call_fn_49385inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_activation_layer_call_and_return_conditional_losses_49398inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_max_pooling2d_6_layer_call_fn_49403inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_49408inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv2d_7_layer_call_fn_49417inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv2d_7_layer_call_and_return_conditional_losses_49436inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѓB№
5__inference_batch_normalization_1_layer_call_fn_49449inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
5__inference_batch_normalization_1_layer_call_fn_49462inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49480inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49498inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_activation_1_layer_call_fn_49503inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_activation_1_layer_call_and_return_conditional_losses_49516inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_max_pooling2d_7_layer_call_fn_49521inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_49526inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv2d_8_layer_call_fn_49535inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_49554inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѓB№
5__inference_batch_normalization_2_layer_call_fn_49567inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
5__inference_batch_normalization_2_layer_call_fn_49580inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49598inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49616inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_activation_2_layer_call_fn_49621inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_activation_2_layer_call_and_return_conditional_losses_49634inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_max_pooling2d_8_layer_call_fn_49639inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_49644inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv2d_9_layer_call_fn_49653inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_49672inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѓB№
5__inference_batch_normalization_3_layer_call_fn_49685inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
5__inference_batch_normalization_3_layer_call_fn_49698inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49716inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49734inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_activation_3_layer_call_fn_49739inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_activation_3_layer_call_and_return_conditional_losses_49752inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv2d_10_layer_call_fn_49761inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv2d_10_layer_call_and_return_conditional_losses_49780inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Џ0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѓB№
5__inference_batch_normalization_4_layer_call_fn_49793inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
5__inference_batch_normalization_4_layer_call_fn_49806inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49824inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49842inputs"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_activation_4_layer_call_fn_49847inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_activation_4_layer_call_and_return_conditional_losses_49860inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_max_pooling2d_9_layer_call_fn_49865inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_49870inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_flatten_2_layer_call_fn_49875inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_flatten_2_layer_call_and_return_conditional_losses_49881inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_dense_7_layer_call_fn_49890inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_dense_7_layer_call_and_return_conditional_losses_49909inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
)__inference_dropout_4_layer_call_fn_49914inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
пBм
)__inference_dropout_4_layer_call_fn_49919inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
D__inference_dropout_4_layer_call_and_return_conditional_losses_49931inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
D__inference_dropout_4_layer_call_and_return_conditional_losses_49936inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_dense_8_layer_call_fn_49945inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_dense_8_layer_call_and_return_conditional_losses_49964inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
)__inference_dropout_5_layer_call_fn_49969inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
пBм
)__inference_dropout_5_layer_call_fn_49974inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
D__inference_dropout_5_layer_call_and_return_conditional_losses_49986inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
D__inference_dropout_5_layer_call_and_return_conditional_losses_49991inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_dense_6_layer_call_fn_50000inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_dense_6_layer_call_and_return_conditional_losses_50011inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
э	variables
ю	keras_api

яtotal

№count"
_tf_keras_metric
c
ё	variables
ђ	keras_api

ѓtotal

єcount
ѕ
_fn_kwargs"
_tf_keras_metric
.:,2Adam/m/conv2d_6/kernel
.:,2Adam/v/conv2d_6/kernel
 :2Adam/m/conv2d_6/bias
 :2Adam/v/conv2d_6/bias
,:*2 Adam/m/batch_normalization/gamma
,:*2 Adam/v/batch_normalization/gamma
+:)2Adam/m/batch_normalization/beta
+:)2Adam/v/batch_normalization/beta
.:,2Adam/m/conv2d_7/kernel
.:,2Adam/v/conv2d_7/kernel
 :2Adam/m/conv2d_7/bias
 :2Adam/v/conv2d_7/bias
.:,2"Adam/m/batch_normalization_1/gamma
.:,2"Adam/v/batch_normalization_1/gamma
-:+2!Adam/m/batch_normalization_1/beta
-:+2!Adam/v/batch_normalization_1/beta
.:, 2Adam/m/conv2d_8/kernel
.:, 2Adam/v/conv2d_8/kernel
 : 2Adam/m/conv2d_8/bias
 : 2Adam/v/conv2d_8/bias
.:, 2"Adam/m/batch_normalization_2/gamma
.:, 2"Adam/v/batch_normalization_2/gamma
-:+ 2!Adam/m/batch_normalization_2/beta
-:+ 2!Adam/v/batch_normalization_2/beta
.:,  2Adam/m/conv2d_9/kernel
.:,  2Adam/v/conv2d_9/kernel
 : 2Adam/m/conv2d_9/bias
 : 2Adam/v/conv2d_9/bias
.:, 2"Adam/m/batch_normalization_3/gamma
.:, 2"Adam/v/batch_normalization_3/gamma
-:+ 2!Adam/m/batch_normalization_3/beta
-:+ 2!Adam/v/batch_normalization_3/beta
/:- 2Adam/m/conv2d_10/kernel
/:- 2Adam/v/conv2d_10/kernel
!:2Adam/m/conv2d_10/bias
!:2Adam/v/conv2d_10/bias
.:,2"Adam/m/batch_normalization_4/gamma
.:,2"Adam/v/batch_normalization_4/gamma
-:+2!Adam/m/batch_normalization_4/beta
-:+2!Adam/v/batch_normalization_4/beta
&:$	2Adam/m/dense_7/kernel
&:$	2Adam/v/dense_7/kernel
 :2Adam/m/dense_7/bias
 :2Adam/v/dense_7/bias
':%
2Adam/m/dense_8/kernel
':%
2Adam/v/dense_8/kernel
 :2Adam/m/dense_8/bias
 :2Adam/v/dense_8/bias
&:$	2Adam/m/dense_6/kernel
&:$	2Adam/v/dense_6/kernel
:2Adam/m/dense_6/bias
:2Adam/v/dense_6/bias
0
я0
№1"
trackable_list_wrapper
.
э	variables"
_generic_user_object
:  (2total
:  (2count
0
ѓ0
є1"
trackable_list_wrapper
.
ё	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
NbL
beta:0B__inference_dense_8_layer_call_and_return_conditional_losses_49964
QbO
	BiasAdd:0B__inference_dense_8_layer_call_and_return_conditional_losses_49964
NbL
beta:0B__inference_dense_8_layer_call_and_return_conditional_losses_48787
QbO
	BiasAdd:0B__inference_dense_8_layer_call_and_return_conditional_losses_48787
NbL
beta:0B__inference_dense_7_layer_call_and_return_conditional_losses_49909
QbO
	BiasAdd:0B__inference_dense_7_layer_call_and_return_conditional_losses_49909
NbL
beta:0B__inference_dense_7_layer_call_and_return_conditional_losses_48750
QbO
	BiasAdd:0B__inference_dense_7_layer_call_and_return_conditional_losses_48750
SbQ
beta:0G__inference_activation_4_layer_call_and_return_conditional_losses_49860
UbS
inputs:0G__inference_activation_4_layer_call_and_return_conditional_losses_49860
SbQ
beta:0G__inference_activation_4_layer_call_and_return_conditional_losses_48722
UbS
inputs:0G__inference_activation_4_layer_call_and_return_conditional_losses_48722
PbN
beta:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_49780
SbQ
	BiasAdd:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_49780
PbN
beta:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_48695
SbQ
	BiasAdd:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_48695
SbQ
beta:0G__inference_activation_3_layer_call_and_return_conditional_losses_49752
UbS
inputs:0G__inference_activation_3_layer_call_and_return_conditional_losses_49752
SbQ
beta:0G__inference_activation_3_layer_call_and_return_conditional_losses_48675
UbS
inputs:0G__inference_activation_3_layer_call_and_return_conditional_losses_48675
ObM
beta:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_49672
RbP
	BiasAdd:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_49672
ObM
beta:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_48648
RbP
	BiasAdd:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_48648
SbQ
beta:0G__inference_activation_2_layer_call_and_return_conditional_losses_49634
UbS
inputs:0G__inference_activation_2_layer_call_and_return_conditional_losses_49634
SbQ
beta:0G__inference_activation_2_layer_call_and_return_conditional_losses_48627
UbS
inputs:0G__inference_activation_2_layer_call_and_return_conditional_losses_48627
ObM
beta:0C__inference_conv2d_8_layer_call_and_return_conditional_losses_49554
RbP
	BiasAdd:0C__inference_conv2d_8_layer_call_and_return_conditional_losses_49554
ObM
beta:0C__inference_conv2d_8_layer_call_and_return_conditional_losses_48600
RbP
	BiasAdd:0C__inference_conv2d_8_layer_call_and_return_conditional_losses_48600
SbQ
beta:0G__inference_activation_1_layer_call_and_return_conditional_losses_49516
UbS
inputs:0G__inference_activation_1_layer_call_and_return_conditional_losses_49516
SbQ
beta:0G__inference_activation_1_layer_call_and_return_conditional_losses_48579
UbS
inputs:0G__inference_activation_1_layer_call_and_return_conditional_losses_48579
ObM
beta:0C__inference_conv2d_7_layer_call_and_return_conditional_losses_49436
RbP
	BiasAdd:0C__inference_conv2d_7_layer_call_and_return_conditional_losses_49436
ObM
beta:0C__inference_conv2d_7_layer_call_and_return_conditional_losses_48552
RbP
	BiasAdd:0C__inference_conv2d_7_layer_call_and_return_conditional_losses_48552
QbO
beta:0E__inference_activation_layer_call_and_return_conditional_losses_49398
SbQ
inputs:0E__inference_activation_layer_call_and_return_conditional_losses_49398
QbO
beta:0E__inference_activation_layer_call_and_return_conditional_losses_48531
SbQ
inputs:0E__inference_activation_layer_call_and_return_conditional_losses_48531
ObM
beta:0C__inference_conv2d_6_layer_call_and_return_conditional_losses_49318
RbP
	BiasAdd:0C__inference_conv2d_6_layer_call_and_return_conditional_losses_49318
ObM
beta:0C__inference_conv2d_6_layer_call_and_return_conditional_losses_48504
RbP
	BiasAdd:0C__inference_conv2d_6_layer_call_and_return_conditional_losses_48504
Bb@
sequential_2/conv2d_6/beta:0 __inference__wrapped_model_48133
EbC
sequential_2/conv2d_6/BiasAdd:0 __inference__wrapped_model_48133
DbB
sequential_2/activation/beta:0 __inference__wrapped_model_48133
YbW
3sequential_2/batch_normalization/FusedBatchNormV3:0 __inference__wrapped_model_48133
Bb@
sequential_2/conv2d_7/beta:0 __inference__wrapped_model_48133
EbC
sequential_2/conv2d_7/BiasAdd:0 __inference__wrapped_model_48133
FbD
 sequential_2/activation_1/beta:0 __inference__wrapped_model_48133
[bY
5sequential_2/batch_normalization_1/FusedBatchNormV3:0 __inference__wrapped_model_48133
Bb@
sequential_2/conv2d_8/beta:0 __inference__wrapped_model_48133
EbC
sequential_2/conv2d_8/BiasAdd:0 __inference__wrapped_model_48133
FbD
 sequential_2/activation_2/beta:0 __inference__wrapped_model_48133
[bY
5sequential_2/batch_normalization_2/FusedBatchNormV3:0 __inference__wrapped_model_48133
Bb@
sequential_2/conv2d_9/beta:0 __inference__wrapped_model_48133
EbC
sequential_2/conv2d_9/BiasAdd:0 __inference__wrapped_model_48133
FbD
 sequential_2/activation_3/beta:0 __inference__wrapped_model_48133
[bY
5sequential_2/batch_normalization_3/FusedBatchNormV3:0 __inference__wrapped_model_48133
CbA
sequential_2/conv2d_10/beta:0 __inference__wrapped_model_48133
FbD
 sequential_2/conv2d_10/BiasAdd:0 __inference__wrapped_model_48133
FbD
 sequential_2/activation_4/beta:0 __inference__wrapped_model_48133
[bY
5sequential_2/batch_normalization_4/FusedBatchNormV3:0 __inference__wrapped_model_48133
Ab?
sequential_2/dense_7/beta:0 __inference__wrapped_model_48133
DbB
sequential_2/dense_7/BiasAdd:0 __inference__wrapped_model_48133
Ab?
sequential_2/dense_8/beta:0 __inference__wrapped_model_48133
DbB
sequential_2/dense_8/BiasAdd:0 __inference__wrapped_model_48133Ъ
 __inference__wrapped_model_48133Ѕ6)*3456IJSTUVijstuvЃЄ­ЎЏАЩЪийчш8Ђ5
.Ђ+
)&
input_3џџџџџџџџџpp
Њ "1Њ.
,
dense_6!
dense_6џџџџџџџџџК
G__inference_activation_1_layer_call_and_return_conditional_losses_49516o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ..
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ..
 
,__inference_activation_1_layer_call_fn_49503d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ..
Њ ")&
unknownџџџџџџџџџ..К
G__inference_activation_2_layer_call_and_return_conditional_losses_49634o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
,__inference_activation_2_layer_call_fn_49621d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџ К
G__inference_activation_3_layer_call_and_return_conditional_losses_49752o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
,__inference_activation_3_layer_call_fn_49739d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџ К
G__inference_activation_4_layer_call_and_return_conditional_losses_49860o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
,__inference_activation_4_layer_call_fn_49847d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџИ
E__inference_activation_layer_call_and_return_conditional_losses_49398o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџff
Њ "4Ђ1
*'
tensor_0џџџџџџџџџff
 
*__inference_activation_layer_call_fn_49385d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџff
Њ ")&
unknownџџџџџџџџџffі
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49480ЁSTUVQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 і
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49498ЁSTUVQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
5__inference_batch_normalization_1_layer_call_fn_49449STUVQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџа
5__inference_batch_normalization_1_layer_call_fn_49462STUVQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџі
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49598ЁstuvQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 і
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_49616ЁstuvQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 а
5__inference_batch_normalization_2_layer_call_fn_49567stuvQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ а
5__inference_batch_normalization_2_layer_call_fn_49580stuvQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ њ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49716ЅQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 њ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49734ЅQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 д
5__inference_batch_normalization_3_layer_call_fn_49685QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
5__inference_batch_normalization_3_layer_call_fn_49698QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ њ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49824Ѕ­ЎЏАQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 њ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49842Ѕ­ЎЏАQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_batch_normalization_4_layer_call_fn_49793­ЎЏАQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
5__inference_batch_normalization_4_layer_call_fn_49806­ЎЏАQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџє
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49362Ё3456QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 є
N__inference_batch_normalization_layer_call_and_return_conditional_losses_49380Ё3456QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
3__inference_batch_normalization_layer_call_fn_493313456QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџЮ
3__inference_batch_normalization_layer_call_fn_493443456QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџН
D__inference_conv2d_10_layer_call_and_return_conditional_losses_49780uЃЄ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
)__inference_conv2d_10_layer_call_fn_49761jЃЄ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџК
C__inference_conv2d_6_layer_call_and_return_conditional_losses_49318s)*7Ђ4
-Ђ*
(%
inputsџџџџџџџџџpp
Њ "4Ђ1
*'
tensor_0џџџџџџџџџff
 
(__inference_conv2d_6_layer_call_fn_49299h)*7Ђ4
-Ђ*
(%
inputsџџџџџџџџџpp
Њ ")&
unknownџџџџџџџџџffК
C__inference_conv2d_7_layer_call_and_return_conditional_losses_49436sIJ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ..
 
(__inference_conv2d_7_layer_call_fn_49417hIJ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ ")&
unknownџџџџџџџџџ..К
C__inference_conv2d_8_layer_call_and_return_conditional_losses_49554sij7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
(__inference_conv2d_8_layer_call_fn_49535hij7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџ М
C__inference_conv2d_9_layer_call_and_return_conditional_losses_49672u7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
(__inference_conv2d_9_layer_call_fn_49653j7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		 
Њ ")&
unknownџџџџџџџџџ Ќ
B__inference_dense_6_layer_call_and_return_conditional_losses_50011fчш0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
'__inference_dense_6_layer_call_fn_50000[чш0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЌ
B__inference_dense_7_layer_call_and_return_conditional_losses_49909fЩЪ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
'__inference_dense_7_layer_call_fn_49890[ЩЪ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ­
B__inference_dense_8_layer_call_and_return_conditional_losses_49964gий0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
'__inference_dense_8_layer_call_fn_49945\ий0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ­
D__inference_dropout_4_layer_call_and_return_conditional_losses_49931e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 ­
D__inference_dropout_4_layer_call_and_return_conditional_losses_49936e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
)__inference_dropout_4_layer_call_fn_49914Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџ
)__inference_dropout_4_layer_call_fn_49919Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџ­
D__inference_dropout_5_layer_call_and_return_conditional_losses_49986e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 ­
D__inference_dropout_5_layer_call_and_return_conditional_losses_49991e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
)__inference_dropout_5_layer_call_fn_49969Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџ
)__inference_dropout_5_layer_call_fn_49974Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџЏ
D__inference_flatten_2_layer_call_and_return_conditional_losses_49881g7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
)__inference_flatten_2_layer_call_fn_49875\7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџё
"__inference_internal_grad_fn_50229ЪіїЂ}
vЂs

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "?<

 
# 
tensor_1џџџџџџџџџ

tensor_2 ё
"__inference_internal_grad_fn_50256ЪјљЂ}
vЂs

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "?<

 
# 
tensor_1џџџџџџџџџ

tensor_2 ё
"__inference_internal_grad_fn_50283ЪњћЂ}
vЂs

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "?<

 
# 
tensor_1џџџџџџџџџ

tensor_2 ё
"__inference_internal_grad_fn_50310Ъќ§Ђ}
vЂs

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "?<

 
# 
tensor_1џџџџџџџџџ

tensor_2 
"__inference_internal_grad_fn_50337тўџЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ
0-
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ

tensor_2 
"__inference_internal_grad_fn_50364тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ
0-
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ

tensor_2 
"__inference_internal_grad_fn_50391тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ
0-
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ

tensor_2 
"__inference_internal_grad_fn_50418тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ
0-
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ

tensor_2 
"__inference_internal_grad_fn_50445тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50472тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50499тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50526тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50553тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50580тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50607тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50634тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_50661тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ..
0-
result_grads_1џџџџџџџџџ..

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ..

tensor_2 
"__inference_internal_grad_fn_50688тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ..
0-
result_grads_1џџџџџџџџџ..

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ..

tensor_2 
"__inference_internal_grad_fn_50715тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ..
0-
result_grads_1џџџџџџџџџ..

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ..

tensor_2 
"__inference_internal_grad_fn_50742тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ..
0-
result_grads_1џџџџџџџџџ..

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ..

tensor_2 
"__inference_internal_grad_fn_50769тЂ
Ђ

 
0-
result_grads_0џџџџџџџџџff
0-
result_grads_1џџџџџџџџџff

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџff

tensor_2 
"__inference_internal_grad_fn_50796т ЁЂ
Ђ

 
0-
result_grads_0џџџџџџџџџff
0-
result_grads_1џџџџџџџџџff

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџff

tensor_2 
"__inference_internal_grad_fn_50823тЂЃЂ
Ђ

 
0-
result_grads_0џџџџџџџџџff
0-
result_grads_1џџџџџџџџџff

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџff

tensor_2 
"__inference_internal_grad_fn_50850тЄЅЂ
Ђ

 
0-
result_grads_0џџџџџџџџџff
0-
result_grads_1џџџџџџџџџff

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџff

tensor_2 
"__inference_internal_grad_fn_50877тІЇЂ
Ђ

 
0-
result_grads_0џџџџџџџџџff
0-
result_grads_1џџџџџџџџџff

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџff

tensor_2 
"__inference_internal_grad_fn_50904тЈЉЂ
Ђ

 
0-
result_grads_0џџџџџџџџџff
0-
result_grads_1џџџџџџџџџff

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџff

tensor_2 
"__inference_internal_grad_fn_50931тЊЋЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ..
0-
result_grads_1џџџџџџџџџ..

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ..

tensor_2 
"__inference_internal_grad_fn_50958тЌ­Ђ
Ђ

 
0-
result_grads_0џџџџџџџџџ..
0-
result_grads_1џџџџџџџџџ..

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ..

tensor_2 
"__inference_internal_grad_fn_50985тЎЏЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_51012тАБЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_51039тВГЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_51066тДЕЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ 
0-
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ 

tensor_2 
"__inference_internal_grad_fn_51093тЖЗЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ
0-
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ

tensor_2 
"__inference_internal_grad_fn_51120тИЙЂ
Ђ

 
0-
result_grads_0џџџџџџџџџ
0-
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "FC

 
*'
tensor_1џџџџџџџџџ

tensor_2 ё
"__inference_internal_grad_fn_51147ЪКЛЂ}
vЂs

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "?<

 
# 
tensor_1џџџџџџџџџ

tensor_2 ё
"__inference_internal_grad_fn_51174ЪМНЂ}
vЂs

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "?<

 
# 
tensor_1џџџџџџџџџ

tensor_2 є
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_49408ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_6_layer_call_fn_49403RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_49526ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_7_layer_call_fn_49521RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_49644ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_8_layer_call_fn_49639RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_49870ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_9_layer_call_fn_49865RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
G__inference_sequential_2_layer_call_and_return_conditional_losses_48823Ј6)*3456IJSTUVijstuvЃЄ­ЎЏАЩЪийчш@Ђ=
6Ђ3
)&
input_3џџџџџџџџџpp
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 є
G__inference_sequential_2_layer_call_and_return_conditional_losses_48934Ј6)*3456IJSTUVijstuvЃЄ­ЎЏАЩЪийчш@Ђ=
6Ђ3
)&
input_3џџџџџџџџџpp
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ю
,__inference_sequential_2_layer_call_fn_490116)*3456IJSTUVijstuvЃЄ­ЎЏАЩЪийчш@Ђ=
6Ђ3
)&
input_3џџџџџџџџџpp
p

 
Њ "!
unknownџџџџџџџџџЮ
,__inference_sequential_2_layer_call_fn_490886)*3456IJSTUVijstuvЃЄ­ЎЏАЩЪийчш@Ђ=
6Ђ3
)&
input_3џџџџџџџџџpp
p 

 
Њ "!
unknownџџџџџџџџџи
#__inference_signature_wrapper_49290А6)*3456IJSTUVijstuvЃЄ­ЎЏАЩЪийчшCЂ@
Ђ 
9Њ6
4
input_3)&
input_3џџџџџџџџџpp"1Њ.
,
dense_6!
dense_6џџџџџџџџџ