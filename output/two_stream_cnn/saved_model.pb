ла
П▐
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
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
╓
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

$
DisableCopyOnRead
resourceИ
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
В
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
└
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8й■
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
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/v/dense_3/kernel
А
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes
:	А*
dtype0
З
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/m/dense_3/kernel
А
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes
:	А*
dtype0

Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_6/bias
x
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_6/bias
x
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_6/kernel
Б
)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_6/kernel
Б
)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel* 
_output_shapes
:
АА*
dtype0

Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_5/bias
x
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_5/bias
x
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_5/kernel
Б
)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_5/kernel
Б
)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel* 
_output_shapes
:
АА*
dtype0

Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_4/bias
x
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_4/bias
x
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_4/kernel
Б
)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_4/kernel
Б
)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel* 
_output_shapes
:
АА*
dtype0
Б
Adam/v/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv2d_5/bias
z
(Adam/v/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv2d_5/bias
z
(Adam/m/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/bias*
_output_shapes	
:А*
dtype0
Т
Adam/v/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/v/conv2d_5/kernel
Л
*Adam/v/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/kernel*(
_output_shapes
:АА*
dtype0
Т
Adam/m/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/m/conv2d_5/kernel
Л
*Adam/m/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/kernel*(
_output_shapes
:АА*
dtype0
Б
Adam/v/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv3d_2/bias
z
(Adam/v/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv3d_2/bias
z
(Adam/m/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/bias*
_output_shapes	
:А*
dtype0
Ц
Adam/v/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:АА*'
shared_nameAdam/v/conv3d_2/kernel
П
*Adam/v/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/kernel*,
_output_shapes
:АА*
dtype0
Ц
Adam/m/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:АА*'
shared_nameAdam/m/conv3d_2/kernel
П
*Adam/m/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/kernel*,
_output_shapes
:АА*
dtype0
Б
Adam/v/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv2d_4/bias
z
(Adam/v/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv2d_4/bias
z
(Adam/m/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/bias*
_output_shapes	
:А*
dtype0
С
Adam/v/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/v/conv2d_4/kernel
К
*Adam/v/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/kernel*'
_output_shapes
:@А*
dtype0
С
Adam/m/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/m/conv2d_4/kernel
К
*Adam/m/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/kernel*'
_output_shapes
:@А*
dtype0
Б
Adam/v/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv3d_1/bias
z
(Adam/v/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv3d_1/bias
z
(Adam/m/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/bias*
_output_shapes	
:А*
dtype0
Х
Adam/v/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*'
shared_nameAdam/v/conv3d_1/kernel
О
*Adam/v/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/kernel*+
_output_shapes
:@А*
dtype0
Х
Adam/m/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*'
shared_nameAdam/m/conv3d_1/kernel
О
*Adam/m/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/kernel*+
_output_shapes
:@А*
dtype0
А
Adam/v/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_3/bias
y
(Adam/v/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_3/bias
y
(Adam/m/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_3/kernel
Й
*Adam/v/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/kernel*&
_output_shapes
:@*
dtype0
Р
Adam/m/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_3/kernel
Й
*Adam/m/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/kernel*&
_output_shapes
:@*
dtype0
|
Adam/v/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/v/conv3d/bias
u
&Adam/v/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/bias*
_output_shapes
:@*
dtype0
|
Adam/m/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/m/conv3d/bias
u
&Adam/m/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv3d/kernel
Й
(Adam/v/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/kernel**
_output_shapes
:@*
dtype0
Р
Adam/m/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv3d/kernel
Й
(Adam/m/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/kernel**
_output_shapes
:@*
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
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	А*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:А*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:А*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:А*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
АА*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:АА*
dtype0
s
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv3d_2/bias
l
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes	
:А*
dtype0
И
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:АА* 
shared_nameconv3d_2/kernel
Б
#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel*,
_output_shapes
:АА*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:А*
dtype0
Г
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_4/kernel
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:@А*
dtype0
s
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv3d_1/bias
l
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes	
:А*
dtype0
З
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А* 
shared_nameconv3d_1/kernel
А
#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel*+
_output_shapes
:@А*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:@*
dtype0
В
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:@*
dtype0
Т
serving_default_input_2Placeholder*3
_output_shapes!
:         
0@*
dtype0*(
shape:         
0@
К
serving_default_input_3Placeholder*/
_output_shapes
:         pp*
dtype0*$
shape:         pp
з
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_3conv3d/kernelconv3d/biasconv2d_3/kernelconv2d_3/biasconv3d_1/kernelconv3d_1/biasconv2d_4/kernelconv2d_4/biasconv3d_2/kernelconv3d_2/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_3/kerneldense_3/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_11788

NoOpNoOp
Мд
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╞г
value╗гB╖г Bпг
Ц
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
 	optimizer
!
signatures*
* 
* 
╚
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op*
╚
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias
 3_jit_compiled_convolution_op*
О
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
О
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
╚
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op*
╚
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op*
О
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses* 
О
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
╚
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op*
╚
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias
 o_jit_compiled_convolution_op*
О
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
О
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|	keras_api* 
С
}	variables
~trainable_variables
regularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses* 
Ф
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses* 
о
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses
Пkernel
	Рbias*
м
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator* 
о
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Юkernel
	Яbias*
м
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
ж_random_generator* 
о
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
нkernel
	оbias*
м
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
│__call__
+┤&call_and_return_all_conditional_losses
╡_random_generator* 
о
╢	variables
╖trainable_variables
╕regularization_losses
╣	keras_api
║__call__
+╗&call_and_return_all_conditional_losses
╝kernel
	╜bias*
в
(0
)1
12
23
F4
G5
O6
P7
d8
e9
m10
n11
П12
Р13
Ю14
Я15
н16
о17
╝18
╜19*
в
(0
)1
12
23
F4
G5
O6
P7
d8
e9
m10
n11
П12
Р13
Ю14
Я15
н16
о17
╝18
╜19*
* 
╡
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

├trace_0
─trace_1* 

┼trace_0
╞trace_1* 
* 
И
╟
_variables
╚_iterations
╔_learning_rate
╩_index_dict
╦
_momentums
╠_velocities
═_update_step_xla*

╬serving_default* 

(0
)1*

(0
)1*
* 
Ш
╧non_trainable_variables
╨layers
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

╘trace_0* 

╒trace_0* 
]W
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

10
21*

10
21*
* 
Ш
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

█trace_0* 

▄trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

тtrace_0* 

уtrace_0* 
* 
* 
* 
Ц
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 

F0
G1*

F0
G1*
* 
Ш
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
_Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

O0
P1*

O0
P1*
* 
Ш
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

ўtrace_0* 

°trace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

■trace_0* 

 trace_0* 
* 
* 
* 
Ц
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 

d0
e1*

d0
e1*
* 
Ш
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
_Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

m0
n1*

m0
n1*
* 
Ш
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 
* 
* 
* 
Ц
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

бtrace_0* 

вtrace_0* 
* 
* 
* 
* 
Щ
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
}	variables
~trainable_variables
regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses* 

иtrace_0* 

йtrace_0* 
* 
* 
* 
Ь
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses* 

пtrace_0* 

░trace_0* 

П0
Р1*

П0
Р1*
* 
Ю
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses*

╢trace_0* 

╖trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

╜trace_0
╛trace_1* 

┐trace_0
└trace_1* 
* 

Ю0
Я1*

Ю0
Я1*
* 
Ю
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses*

╞trace_0* 

╟trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses* 

═trace_0
╬trace_1* 

╧trace_0
╨trace_1* 
* 

н0
о1*

н0
о1*
* 
Ю
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses*

╓trace_0* 

╫trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╪non_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
п	variables
░trainable_variables
▒regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses* 

▌trace_0
▐trace_1* 

▀trace_0
рtrace_1* 
* 

╝0
╜1*

╝0
╜1*
* 
Ю
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
╢	variables
╖trainable_variables
╕regularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses*

цtrace_0* 

чtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
║
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
23*

ш0
щ1*
* 
* 
* 
* 
* 
* 
ы
╚0
ъ1
ы2
ь3
э4
ю5
я6
Ё7
ё8
Є9
є10
Ї11
ї12
Ў13
ў14
°15
∙16
·17
√18
№19
¤20
■21
 22
А23
Б24
В25
Г26
Д27
Е28
Ж29
З30
И31
Й32
К33
Л34
М35
Н36
О37
П38
Р39
С40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
о
ъ0
ь1
ю2
Ё3
Є4
Ї5
Ў6
°7
·8
№9
■10
А11
В12
Д13
Ж14
И15
К16
М17
О18
Р19*
о
ы0
э1
я2
ё3
є4
ї5
ў6
∙7
√8
¤9
 10
Б11
Г12
Е13
З14
Й15
Л16
Н17
П18
С19*
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
Т	variables
У	keras_api

Фtotal

Хcount*
M
Ц	variables
Ч	keras_api

Шtotal

Щcount
Ъ
_fn_kwargs*
_Y
VARIABLE_VALUEAdam/m/conv3d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv3d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv3d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_3/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_3/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_3/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_3/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_4/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_4/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_4/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_4/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_5/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_5/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_5/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_5/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_4/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_4/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_4/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_5/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_5/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_5/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_5/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_6/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_6/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_6/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_6/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*

Ф0
Х1*

Т	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ш0
Щ1*

Ц	variables*
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
╩
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv2d_3/kernelconv2d_3/biasconv3d_1/kernelconv3d_1/biasconv2d_4/kernelconv2d_4/biasconv3d_2/kernelconv3d_2/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_3/kerneldense_3/bias	iterationlearning_rateAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/biasAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/biasAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/biasAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/biasAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/biasAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotal_1count_1totalcountConst*O
TinH
F2D*
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
GPU 2J 8В *'
f"R 
__inference__traced_save_13106
┼
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv2d_3/kernelconv2d_3/biasconv3d_1/kernelconv3d_1/biasconv2d_4/kernelconv2d_4/biasconv3d_2/kernelconv3d_2/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_3/kerneldense_3/bias	iterationlearning_rateAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/biasAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/biasAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/biasAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/biasAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/biasAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotal_1count_1totalcount*N
TinG
E2C*
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_13313▓╢
┤Є
Б=
__inference__traced_save_13106
file_prefixB
$read_disablecopyonread_conv3d_kernel:@2
$read_1_disablecopyonread_conv3d_bias:@B
(read_2_disablecopyonread_conv2d_3_kernel:@4
&read_3_disablecopyonread_conv2d_3_bias:@G
(read_4_disablecopyonread_conv3d_1_kernel:@А5
&read_5_disablecopyonread_conv3d_1_bias:	АC
(read_6_disablecopyonread_conv2d_4_kernel:@А5
&read_7_disablecopyonread_conv2d_4_bias:	АH
(read_8_disablecopyonread_conv3d_2_kernel:АА5
&read_9_disablecopyonread_conv3d_2_bias:	АE
)read_10_disablecopyonread_conv2d_5_kernel:АА6
'read_11_disablecopyonread_conv2d_5_bias:	А<
(read_12_disablecopyonread_dense_4_kernel:
АА5
&read_13_disablecopyonread_dense_4_bias:	А<
(read_14_disablecopyonread_dense_5_kernel:
АА5
&read_15_disablecopyonread_dense_5_bias:	А<
(read_16_disablecopyonread_dense_6_kernel:
АА5
&read_17_disablecopyonread_dense_6_bias:	А;
(read_18_disablecopyonread_dense_3_kernel:	А4
&read_19_disablecopyonread_dense_3_bias:-
#read_20_disablecopyonread_iteration:	 1
'read_21_disablecopyonread_learning_rate: L
.read_22_disablecopyonread_adam_m_conv3d_kernel:@L
.read_23_disablecopyonread_adam_v_conv3d_kernel:@:
,read_24_disablecopyonread_adam_m_conv3d_bias:@:
,read_25_disablecopyonread_adam_v_conv3d_bias:@J
0read_26_disablecopyonread_adam_m_conv2d_3_kernel:@J
0read_27_disablecopyonread_adam_v_conv2d_3_kernel:@<
.read_28_disablecopyonread_adam_m_conv2d_3_bias:@<
.read_29_disablecopyonread_adam_v_conv2d_3_bias:@O
0read_30_disablecopyonread_adam_m_conv3d_1_kernel:@АO
0read_31_disablecopyonread_adam_v_conv3d_1_kernel:@А=
.read_32_disablecopyonread_adam_m_conv3d_1_bias:	А=
.read_33_disablecopyonread_adam_v_conv3d_1_bias:	АK
0read_34_disablecopyonread_adam_m_conv2d_4_kernel:@АK
0read_35_disablecopyonread_adam_v_conv2d_4_kernel:@А=
.read_36_disablecopyonread_adam_m_conv2d_4_bias:	А=
.read_37_disablecopyonread_adam_v_conv2d_4_bias:	АP
0read_38_disablecopyonread_adam_m_conv3d_2_kernel:ААP
0read_39_disablecopyonread_adam_v_conv3d_2_kernel:АА=
.read_40_disablecopyonread_adam_m_conv3d_2_bias:	А=
.read_41_disablecopyonread_adam_v_conv3d_2_bias:	АL
0read_42_disablecopyonread_adam_m_conv2d_5_kernel:ААL
0read_43_disablecopyonread_adam_v_conv2d_5_kernel:АА=
.read_44_disablecopyonread_adam_m_conv2d_5_bias:	А=
.read_45_disablecopyonread_adam_v_conv2d_5_bias:	АC
/read_46_disablecopyonread_adam_m_dense_4_kernel:
ААC
/read_47_disablecopyonread_adam_v_dense_4_kernel:
АА<
-read_48_disablecopyonread_adam_m_dense_4_bias:	А<
-read_49_disablecopyonread_adam_v_dense_4_bias:	АC
/read_50_disablecopyonread_adam_m_dense_5_kernel:
ААC
/read_51_disablecopyonread_adam_v_dense_5_kernel:
АА<
-read_52_disablecopyonread_adam_m_dense_5_bias:	А<
-read_53_disablecopyonread_adam_v_dense_5_bias:	АC
/read_54_disablecopyonread_adam_m_dense_6_kernel:
ААC
/read_55_disablecopyonread_adam_v_dense_6_kernel:
АА<
-read_56_disablecopyonread_adam_m_dense_6_bias:	А<
-read_57_disablecopyonread_adam_v_dense_6_bias:	АB
/read_58_disablecopyonread_adam_m_dense_3_kernel:	АB
/read_59_disablecopyonread_adam_v_dense_3_kernel:	А;
-read_60_disablecopyonread_adam_m_dense_3_bias:;
-read_61_disablecopyonread_adam_v_dense_3_bias:+
!read_62_disablecopyonread_total_1: +
!read_63_disablecopyonread_count_1: )
read_64_disablecopyonread_total: )
read_65_disablecopyonread_count: 
savev2_const
identity_133ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_31/DisableCopyOnReadвRead_31/ReadVariableOpвRead_32/DisableCopyOnReadвRead_32/ReadVariableOpвRead_33/DisableCopyOnReadвRead_33/ReadVariableOpвRead_34/DisableCopyOnReadвRead_34/ReadVariableOpвRead_35/DisableCopyOnReadвRead_35/ReadVariableOpвRead_36/DisableCopyOnReadвRead_36/ReadVariableOpвRead_37/DisableCopyOnReadвRead_37/ReadVariableOpвRead_38/DisableCopyOnReadвRead_38/ReadVariableOpвRead_39/DisableCopyOnReadвRead_39/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_40/DisableCopyOnReadвRead_40/ReadVariableOpвRead_41/DisableCopyOnReadвRead_41/ReadVariableOpвRead_42/DisableCopyOnReadвRead_42/ReadVariableOpвRead_43/DisableCopyOnReadвRead_43/ReadVariableOpвRead_44/DisableCopyOnReadвRead_44/ReadVariableOpвRead_45/DisableCopyOnReadвRead_45/ReadVariableOpвRead_46/DisableCopyOnReadвRead_46/ReadVariableOpвRead_47/DisableCopyOnReadвRead_47/ReadVariableOpвRead_48/DisableCopyOnReadвRead_48/ReadVariableOpвRead_49/DisableCopyOnReadвRead_49/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_50/DisableCopyOnReadвRead_50/ReadVariableOpвRead_51/DisableCopyOnReadвRead_51/ReadVariableOpвRead_52/DisableCopyOnReadвRead_52/ReadVariableOpвRead_53/DisableCopyOnReadвRead_53/ReadVariableOpвRead_54/DisableCopyOnReadвRead_54/ReadVariableOpвRead_55/DisableCopyOnReadвRead_55/ReadVariableOpвRead_56/DisableCopyOnReadвRead_56/ReadVariableOpвRead_57/DisableCopyOnReadвRead_57/ReadVariableOpвRead_58/DisableCopyOnReadвRead_58/ReadVariableOpвRead_59/DisableCopyOnReadвRead_59/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_60/DisableCopyOnReadвRead_60/ReadVariableOpвRead_61/DisableCopyOnReadвRead_61/ReadVariableOpвRead_62/DisableCopyOnReadвRead_62/ReadVariableOpвRead_63/DisableCopyOnReadвRead_63/ReadVariableOpвRead_64/DisableCopyOnReadвRead_64/ReadVariableOpвRead_65/DisableCopyOnReadвRead_65/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv3d_kernel"/device:CPU:0*
_output_shapes
 м
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv3d_kernel^Read/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0u
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@m

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0**
_output_shapes
:@x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv3d_bias"/device:CPU:0*
_output_shapes
 а
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv3d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 ░
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:@z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 в
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv3d_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*+
_output_shapes
:@А*
dtype0z

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*+
_output_shapes
:@Аp

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*+
_output_shapes
:@Аz
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv3d_1_bias"/device:CPU:0*
_output_shapes
 г
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv3d_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:А|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_4_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0w
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*'
_output_shapes
:@Аz
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 г
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_4_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:А|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 ╢
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_conv3d_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*,
_output_shapes
:АА*
dtype0|
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*,
_output_shapes
:ААs
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*,
_output_shapes
:ААz
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_conv3d_2_bias"/device:CPU:0*
_output_shapes
 г
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_conv3d_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:А~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_conv2d_5_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0y
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*(
_output_shapes
:АА|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 ж
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_conv2d_5_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 м
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_4_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 е
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_4_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 м
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_dense_5_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_15/DisableCopyOnReadDisableCopyOnRead&read_15_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 е
Read_15/ReadVariableOpReadVariableOp&read_15_disablecopyonread_dense_5_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 м
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_dense_6_kernel^Read_16/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 е
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_dense_6_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 л
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_dense_3_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	А{
Read_19/DisableCopyOnReadDisableCopyOnRead&read_19_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 д
Read_19/ReadVariableOpReadVariableOp&read_19_disablecopyonread_dense_3_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_20/DisableCopyOnReadDisableCopyOnRead#read_20_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_20/ReadVariableOpReadVariableOp#read_20_disablecopyonread_iteration^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 б
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_learning_rate^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_m_conv3d_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_m_conv3d_kernel^Read_22/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0{
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@q
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0**
_output_shapes
:@Г
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_adam_v_conv3d_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_adam_v_conv3d_kernel^Read_23/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0{
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@q
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0**
_output_shapes
:@Б
Read_24/DisableCopyOnReadDisableCopyOnRead,read_24_disablecopyonread_adam_m_conv3d_bias"/device:CPU:0*
_output_shapes
 к
Read_24/ReadVariableOpReadVariableOp,read_24_disablecopyonread_adam_m_conv3d_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@Б
Read_25/DisableCopyOnReadDisableCopyOnRead,read_25_disablecopyonread_adam_v_conv3d_bias"/device:CPU:0*
_output_shapes
 к
Read_25/ReadVariableOpReadVariableOp,read_25_disablecopyonread_adam_v_conv3d_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 ║
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_conv2d_3_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Е
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 ║
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_conv2d_3_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Г
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_conv2d_3_bias"/device:CPU:0*
_output_shapes
 м
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_conv2d_3_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_conv2d_3_bias"/device:CPU:0*
_output_shapes
 м
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_conv2d_3_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_conv3d_1_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*+
_output_shapes
:@А*
dtype0|
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*+
_output_shapes
:@Аr
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*+
_output_shapes
:@АЕ
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_conv3d_1_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*+
_output_shapes
:@А*
dtype0|
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*+
_output_shapes
:@Аr
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*+
_output_shapes
:@АГ
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_conv3d_1_bias"/device:CPU:0*
_output_shapes
 н
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_conv3d_1_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_conv3d_1_bias"/device:CPU:0*
_output_shapes
 н
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_conv3d_1_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_conv2d_4_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*'
_output_shapes
:@АЕ
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_conv2d_4_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*'
_output_shapes
:@АГ
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_conv2d_4_bias"/device:CPU:0*
_output_shapes
 н
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_conv2d_4_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_conv2d_4_bias"/device:CPU:0*
_output_shapes
 н
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_conv2d_4_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 └
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_conv3d_2_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*,
_output_shapes
:АА*
dtype0}
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*,
_output_shapes
:ААs
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*,
_output_shapes
:ААЕ
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 └
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_conv3d_2_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*,
_output_shapes
:АА*
dtype0}
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*,
_output_shapes
:ААs
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*,
_output_shapes
:ААГ
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_conv3d_2_bias"/device:CPU:0*
_output_shapes
 н
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_conv3d_2_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_conv3d_2_bias"/device:CPU:0*
_output_shapes
 н
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_conv3d_2_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_conv2d_5_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0y
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААЕ
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_conv2d_5_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0y
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААГ
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_m_conv2d_5_bias"/device:CPU:0*
_output_shapes
 н
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_m_conv2d_5_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_v_conv2d_5_bias"/device:CPU:0*
_output_shapes
 н
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_v_conv2d_5_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_46/DisableCopyOnReadDisableCopyOnRead/read_46_disablecopyonread_adam_m_dense_4_kernel"/device:CPU:0*
_output_shapes
 │
Read_46/ReadVariableOpReadVariableOp/read_46_disablecopyonread_adam_m_dense_4_kernel^Read_46/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_47/DisableCopyOnReadDisableCopyOnRead/read_47_disablecopyonread_adam_v_dense_4_kernel"/device:CPU:0*
_output_shapes
 │
Read_47/ReadVariableOpReadVariableOp/read_47_disablecopyonread_adam_v_dense_4_kernel^Read_47/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_48/DisableCopyOnReadDisableCopyOnRead-read_48_disablecopyonread_adam_m_dense_4_bias"/device:CPU:0*
_output_shapes
 м
Read_48/ReadVariableOpReadVariableOp-read_48_disablecopyonread_adam_m_dense_4_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_49/DisableCopyOnReadDisableCopyOnRead-read_49_disablecopyonread_adam_v_dense_4_bias"/device:CPU:0*
_output_shapes
 м
Read_49/ReadVariableOpReadVariableOp-read_49_disablecopyonread_adam_v_dense_4_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_50/DisableCopyOnReadDisableCopyOnRead/read_50_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 │
Read_50/ReadVariableOpReadVariableOp/read_50_disablecopyonread_adam_m_dense_5_kernel^Read_50/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_51/DisableCopyOnReadDisableCopyOnRead/read_51_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 │
Read_51/ReadVariableOpReadVariableOp/read_51_disablecopyonread_adam_v_dense_5_kernel^Read_51/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_52/DisableCopyOnReadDisableCopyOnRead-read_52_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 м
Read_52/ReadVariableOpReadVariableOp-read_52_disablecopyonread_adam_m_dense_5_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_53/DisableCopyOnReadDisableCopyOnRead-read_53_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 м
Read_53/ReadVariableOpReadVariableOp-read_53_disablecopyonread_adam_v_dense_5_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_54/DisableCopyOnReadDisableCopyOnRead/read_54_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 │
Read_54/ReadVariableOpReadVariableOp/read_54_disablecopyonread_adam_m_dense_6_kernel^Read_54/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_55/DisableCopyOnReadDisableCopyOnRead/read_55_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 │
Read_55/ReadVariableOpReadVariableOp/read_55_disablecopyonread_adam_v_dense_6_kernel^Read_55/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_56/DisableCopyOnReadDisableCopyOnRead-read_56_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 м
Read_56/ReadVariableOpReadVariableOp-read_56_disablecopyonread_adam_m_dense_6_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_57/DisableCopyOnReadDisableCopyOnRead-read_57_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 м
Read_57/ReadVariableOpReadVariableOp-read_57_disablecopyonread_adam_v_dense_6_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_58/DisableCopyOnReadDisableCopyOnRead/read_58_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_58/ReadVariableOpReadVariableOp/read_58_disablecopyonread_adam_m_dense_3_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0q
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аh
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:	АД
Read_59/DisableCopyOnReadDisableCopyOnRead/read_59_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_59/ReadVariableOpReadVariableOp/read_59_disablecopyonread_adam_v_dense_3_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0q
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аh
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_60/DisableCopyOnReadDisableCopyOnRead-read_60_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 л
Read_60/ReadVariableOpReadVariableOp-read_60_disablecopyonread_adam_m_dense_3_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_61/DisableCopyOnReadDisableCopyOnRead-read_61_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 л
Read_61/ReadVariableOpReadVariableOp-read_61_disablecopyonread_adam_v_dense_3_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_62/DisableCopyOnReadDisableCopyOnRead!read_62_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_62/ReadVariableOpReadVariableOp!read_62_disablecopyonread_total_1^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_63/DisableCopyOnReadDisableCopyOnRead!read_63_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_63/ReadVariableOpReadVariableOp!read_63_disablecopyonread_count_1^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_64/DisableCopyOnReadDisableCopyOnReadread_64_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_64/ReadVariableOpReadVariableOpread_64_disablecopyonread_total^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_65/DisableCopyOnReadDisableCopyOnReadread_65_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_65/ReadVariableOpReadVariableOpread_65_disablecopyonread_count^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
: м
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*╒
value╦B╚CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╤
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *Q
dtypesG
E2C	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_132Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_133IdentityIdentity_132:output:0^NoOp*
T0*
_output_shapes
: ╔
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_133Identity_133:output:0*(
_construction_contextkEagerRuntime*Э
_input_shapesЛ
И: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_65/ReadVariableOpRead_65/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=C9

_output_shapes
: 

_user_specified_nameConst:%B!

_user_specified_namecount:%A!

_user_specified_nametotal:'@#
!
_user_specified_name	count_1:'?#
!
_user_specified_name	total_1:3>/
-
_user_specified_nameAdam/v/dense_3/bias:3=/
-
_user_specified_nameAdam/m/dense_3/bias:5<1
/
_user_specified_nameAdam/v/dense_3/kernel:5;1
/
_user_specified_nameAdam/m/dense_3/kernel:3:/
-
_user_specified_nameAdam/v/dense_6/bias:39/
-
_user_specified_nameAdam/m/dense_6/bias:581
/
_user_specified_nameAdam/v/dense_6/kernel:571
/
_user_specified_nameAdam/m/dense_6/kernel:36/
-
_user_specified_nameAdam/v/dense_5/bias:35/
-
_user_specified_nameAdam/m/dense_5/bias:541
/
_user_specified_nameAdam/v/dense_5/kernel:531
/
_user_specified_nameAdam/m/dense_5/kernel:32/
-
_user_specified_nameAdam/v/dense_4/bias:31/
-
_user_specified_nameAdam/m/dense_4/bias:501
/
_user_specified_nameAdam/v/dense_4/kernel:5/1
/
_user_specified_nameAdam/m/dense_4/kernel:4.0
.
_user_specified_nameAdam/v/conv2d_5/bias:4-0
.
_user_specified_nameAdam/m/conv2d_5/bias:6,2
0
_user_specified_nameAdam/v/conv2d_5/kernel:6+2
0
_user_specified_nameAdam/m/conv2d_5/kernel:4*0
.
_user_specified_nameAdam/v/conv3d_2/bias:4)0
.
_user_specified_nameAdam/m/conv3d_2/bias:6(2
0
_user_specified_nameAdam/v/conv3d_2/kernel:6'2
0
_user_specified_nameAdam/m/conv3d_2/kernel:4&0
.
_user_specified_nameAdam/v/conv2d_4/bias:4%0
.
_user_specified_nameAdam/m/conv2d_4/bias:6$2
0
_user_specified_nameAdam/v/conv2d_4/kernel:6#2
0
_user_specified_nameAdam/m/conv2d_4/kernel:4"0
.
_user_specified_nameAdam/v/conv3d_1/bias:4!0
.
_user_specified_nameAdam/m/conv3d_1/bias:6 2
0
_user_specified_nameAdam/v/conv3d_1/kernel:62
0
_user_specified_nameAdam/m/conv3d_1/kernel:40
.
_user_specified_nameAdam/v/conv2d_3/bias:40
.
_user_specified_nameAdam/m/conv2d_3/bias:62
0
_user_specified_nameAdam/v/conv2d_3/kernel:62
0
_user_specified_nameAdam/m/conv2d_3/kernel:2.
,
_user_specified_nameAdam/v/conv3d/bias:2.
,
_user_specified_nameAdam/m/conv3d/bias:40
.
_user_specified_nameAdam/v/conv3d/kernel:40
.
_user_specified_nameAdam/m/conv3d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_4/bias:.*
(
_user_specified_namedense_4/kernel:-)
'
_user_specified_nameconv2d_5/bias:/+
)
_user_specified_nameconv2d_5/kernel:-
)
'
_user_specified_nameconv3d_2/bias:/	+
)
_user_specified_nameconv3d_2/kernel:-)
'
_user_specified_nameconv2d_4/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_3/kernel:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╥

Ї
B__inference_dense_3_layer_call_and_return_conditional_losses_12201

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
у
╗
"__inference_internal_grad_fn_12715
result_grads_0
result_grads_1
result_grads_2
mul_model_conv2d_3_beta
mul_model_conv2d_3_biasadd
identity

identity_1К
mulMulmul_model_conv2d_3_betamul_model_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:         ll@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:         ll@{
mul_1Mulmul_model_conv2d_3_betamul_model_conv2d_3_biasadd*
T0*/
_output_shapes
:         ll@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:         ll@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:         ll@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:         ll@f
SquareSquaremul_model_conv2d_3_biasadd*
T0*/
_output_shapes
:         ll@b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:         ll@^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:         ll@^
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
:         ll@Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:         ll@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ll@:         ll@: : :         ll@:gc
/
_output_shapes
:         ll@
0
_user_specified_namemodel/conv2d_3/BiasAdd:KG

_output_shapes
: 
-
_user_specified_namemodel/conv2d_3/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:_[
/
_output_shapes
:         ll@
(
_user_specified_nameresult_grads_1:И Г
&
 _has_manual_control_dependencies(
/
_output_shapes
:         ll@
(
_user_specified_nameresult_grads_0
Т
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11113

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╪
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12006

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
Я

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_12129

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠
╗
"__inference_internal_grad_fn_12796
result_grads_0
result_grads_1
result_grads_2
mul_model_conv3d_2_beta
mul_model_conv3d_2_biasadd
identity

identity_1П
mulMulmul_model_conv3d_2_betamul_model_conv3d_2_biasadd^result_grads_0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         АА
mul_1Mulmul_model_conv3d_2_betamul_model_conv3d_2_biasadd*
T0*4
_output_shapes"
 :         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
subSubsub/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
addAddV2add/x:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :         Аa
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :         Аk
SquareSquaremul_model_conv3d_2_biasadd*
T0*4
_output_shapes"
 :         Аg
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :         Аc
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         Аa
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :         Аb
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         А:         А: : :         А:lh
4
_output_shapes"
 :         А
0
_user_specified_namemodel/conv3d_2/BiasAdd:KG

_output_shapes
: 
-
_user_specified_namemodel/conv3d_2/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_1:Н И
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_0
ъ
K
/__inference_max_pooling3d_2_layer_call_fn_12001

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A                                             * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11143Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
С\
Й	
@__inference_model_layer_call_and_return_conditional_losses_11516
input_3
input_2*
conv3d_11437:@
conv3d_11439:@(
conv2d_3_11443:@
conv2d_3_11445:@-
conv3d_1_11448:@А
conv3d_1_11450:	А)
conv2d_4_11455:@А
conv2d_4_11457:	А.
conv3d_2_11460:АА
conv3d_2_11462:	А*
conv2d_5_11467:АА
conv2d_5_11469:	А!
dense_4_11477:
АА
dense_4_11479:	А!
dense_5_11488:
АА
dense_5_11490:	А!
dense_6_11499:
АА
dense_6_11501:	А 
dense_3_11510:	А
dense_3_11512:
identityИв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallвdense_6/StatefulPartitionedCallЄ
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv3d_11437conv3d_11439*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         	,<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_11180ю
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         	@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11103Ў
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_3_11443conv2d_3_11445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11205Ъ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_11448conv3d_1_11450*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11229Ё
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11113ї
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11123Ш
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_11455conv2d_4_11457*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11255Ь
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_11460conv3d_2_11462*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11279ё
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11133ї
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11143Ш
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_5_11467conv2d_5_11469*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11305ё
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_11153q
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             е
tf.reshape/ReshapeReshape(max_pooling3d_2/PartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         АЖ
concatenate/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0tf.reshape/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_11320╪
flatten_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_11327Ж
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_11477dense_4_11479*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_11339▄
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_11486Ж
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_5_11488dense_5_11490*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_11368▄
dropout_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_11497Ж
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_6_11499dense_6_11501*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_11397▄
dropout_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_11508Е
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_3_11510dense_3_11512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_11426w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ·
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:%!

_user_specified_name11512:%!

_user_specified_name11510:%!

_user_specified_name11501:%!

_user_specified_name11499:%!

_user_specified_name11490:%!

_user_specified_name11488:%!

_user_specified_name11479:%!

_user_specified_name11477:%!

_user_specified_name11469:%!

_user_specified_name11467:%!

_user_specified_name11462:%
!

_user_specified_name11460:%	!

_user_specified_name11457:%!

_user_specified_name11455:%!

_user_specified_name11450:%!

_user_specified_name11448:%!

_user_specified_name11445:%!

_user_specified_name11443:%!

_user_specified_name11439:%!

_user_specified_name11437:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_2:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_3
├
Э
"__inference_internal_grad_fn_12526
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аb
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
subSubsub/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
addAddV2add/x:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :         Аa
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :         А\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :         Аg
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :         Аc
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         Аa
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :         Аb
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         А:         А: : :         А:]Y
4
_output_shapes"
 :         А
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
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_1:Н И
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_0
ў
╗
"__inference_internal_grad_fn_12823
result_grads_0
result_grads_1
result_grads_2
mul_model_conv2d_5_beta
mul_model_conv2d_5_biasadd
identity

identity_1Л
mulMulmul_model_conv2d_5_betamul_model_conv2d_5_biasadd^result_grads_0*
T0*0
_output_shapes
:         АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:         А|
mul_1Mulmul_model_conv2d_5_betamul_model_conv2d_5_biasadd*
T0*0
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         А[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:         А]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:         Аg
SquareSquaremul_model_conv2d_5_biasadd*
T0*0
_output_shapes
:         Аc
mul_4Mulresult_grads_0
Square:y:0*
T0*0
_output_shapes
:         А_
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*0
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         А]
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*0
_output_shapes
:         А^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: b
mul_7Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:         АZ
IdentityIdentity	mul_7:z:0*
T0*0
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         А:         А: : :         А:hd
0
_output_shapes
:         А
0
_user_specified_namemodel/conv2d_5/BiasAdd:KG

_output_shapes
: 
-
_user_specified_namemodel/conv2d_5/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:`\
0
_output_shapes
:         А
(
_user_specified_nameresult_grads_1:Й Д
&
 _has_manual_control_dependencies(
0
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
█
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_11508

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╥

Ї
B__inference_dense_3_layer_call_and_return_conditional_losses_11426

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
Л
Б
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11305

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:         АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:         Аf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         АZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:         А═
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11296*N
_output_shapes<
::         А:         А: l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         

А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
Я
▀
%__inference_model_layer_call_fn_11562
input_3
input_2%
unknown:@
	unknown_0:@#
	unknown_1:@
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А$
	unknown_5:@А
	unknown_6:	А)
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А

unknown_18:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCallinput_3input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11558:%!

_user_specified_name11556:%!

_user_specified_name11554:%!

_user_specified_name11552:%!

_user_specified_name11550:%!

_user_specified_name11548:%!

_user_specified_name11546:%!

_user_specified_name11544:%!

_user_specified_name11542:%!

_user_specified_name11540:%!

_user_specified_name11538:%
!

_user_specified_name11536:%	!

_user_specified_name11534:%!

_user_specified_name11532:%!

_user_specified_name11530:%!

_user_specified_name11528:%!

_user_specified_name11526:%!

_user_specified_name11524:%!

_user_specified_name11522:%!

_user_specified_name11520:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_2:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_3
к
г
(__inference_conv3d_1_layer_call_fn_11873

inputs&
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11229|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         	@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11869:%!

_user_specified_name11867:[ W
3
_output_shapes!
:         	@
 
_user_specified_nameinputs
Я

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_11414

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
E
)__inference_dropout_4_layer_call_fn_12164

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_11508a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
я
Ч
'__inference_dense_5_layer_call_fn_12096

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_11368p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12092:%!

_user_specified_name12090:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Л
Б
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11996

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:         АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:         Аf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         АZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:         А═
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11987*N
_output_shapes<
::         А:         А: l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         

А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11133

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я
Э
"__inference_internal_grad_fn_12472
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1m
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:           АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:           А^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:           АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           А[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:           АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:           А]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:           АX
SquareSquaremul_biasadd*
T0*0
_output_shapes
:           Аc
mul_4Mulresult_grads_0
Square:y:0*
T0*0
_output_shapes
:           А_
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*0
_output_shapes
:           АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           А]
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*0
_output_shapes
:           А^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: b
mul_7Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:           АZ
IdentityIdentity	mul_7:z:0*
T0*0
_output_shapes
:           АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:           А:           А: : :           А:YU
0
_output_shapes
:           А
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
_user_specified_nameresult_grads_2:`\
0
_output_shapes
:           А
(
_user_specified_nameresult_grads_1:Й Д
&
 _has_manual_control_dependencies(
0
_output_shapes
:           А
(
_user_specified_nameresult_grads_0
Я

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_12176

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╓
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11103

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
я
Ч
'__inference_dense_6_layer_call_fn_12143

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_11397p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12139:%!

_user_specified_name12137:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
E
)__inference_dropout_3_layer_call_fn_12117

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_11497a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┬
Е
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11968

inputs>
conv3d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpВ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:АА*
dtype0а
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аj
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :         А╒
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11959*V
_output_shapesD
B:         А:         А: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :         АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :         А
 
_user_specified_nameinputs
п
А
A__inference_conv3d_layer_call_and_return_conditional_losses_11180

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         	,<@*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         	,<@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
mulMulbeta:output:0BiasAdd:output:0*
T0*3
_output_shapes!
:         	,<@Y
SigmoidSigmoidmul:z:0*
T0*3
_output_shapes!
:         	,<@i
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@]
IdentityIdentity	mul_1:z:0*
T0*3
_output_shapes!
:         	,<@╙
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11171*T
_output_shapesB
@:         	,<@:         	,<@: o

Identity_1IdentityIdentityN:output:0^NoOp*
T0*3
_output_shapes!
:         	,<@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
0@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:         
0@
 
_user_specified_nameinputs
█
Э
"__inference_internal_grad_fn_12580
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
:         ll@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:         ll@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:         ll@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:         ll@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:         ll@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:         ll@W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:         ll@b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:         ll@^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:         ll@^
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
:         ll@Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:         ll@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ll@:         ll@: : :         ll@:XT
/
_output_shapes
:         ll@
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
:         ll@
(
_user_specified_nameresult_grads_1:И Г
&
 _has_manual_control_dependencies(
/
_output_shapes
:         ll@
(
_user_specified_nameresult_grads_0
в
Я
&__inference_conv3d_layer_call_fn_11797

inputs%
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         	,<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_11180{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         	,<@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
0@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11793:%!

_user_specified_name11791:[ W
3
_output_shapes!
:         
0@
 
_user_specified_nameinputs
п
Э
"__inference_internal_grad_fn_12661
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1p
mulMulmul_betamul_biasadd^result_grads_0*
T0*3
_output_shapes!
:         	,<@Y
SigmoidSigmoidmul:z:0*
T0*3
_output_shapes!
:         	,<@a
mul_1Mulmul_betamul_biasadd*
T0*3
_output_shapes!
:         	,<@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
subSubsub/x:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@^
mul_2Mul	mul_1:z:0sub:z:0*
T0*3
_output_shapes!
:         	,<@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
addAddV2add/x:output:0	mul_2:z:0*
T0*3
_output_shapes!
:         	,<@`
mul_3MulSigmoid:y:0add:z:0*
T0*3
_output_shapes!
:         	,<@[
SquareSquaremul_biasadd*
T0*3
_output_shapes!
:         	,<@f
mul_4Mulresult_grads_0
Square:y:0*
T0*3
_output_shapes!
:         	,<@b
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@`
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*3
_output_shapes!
:         	,<@b
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: e
mul_7Mulresult_grads_0	mul_3:z:0*
T0*3
_output_shapes!
:         	,<@]
IdentityIdentity	mul_7:z:0*
T0*3
_output_shapes!
:         	,<@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         	,<@:         	,<@: : :         	,<@:\X
3
_output_shapes!
:         	,<@
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
_user_specified_nameresult_grads_2:c_
3
_output_shapes!
:         	,<@
(
_user_specified_nameresult_grads_1:М З
&
 _has_manual_control_dependencies(
3
_output_shapes!
:         	,<@
(
_user_specified_nameresult_grads_0
╓
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11854

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
╧
b
)__inference_dropout_2_layer_call_fn_12065

inputs
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_11356p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
З
А
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11255

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:           АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:           Аf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           АZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:           А═
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11246*N
_output_shapes<
::           А:           А: l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:           АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$@: : 20
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
:         $$@
 
_user_specified_nameinputs
├
Э
"__inference_internal_grad_fn_12418
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аb
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
subSubsub/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
addAddV2add/x:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :         Аa
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :         А\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :         Аg
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :         Аc
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         Аa
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :         Аb
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         А:         А: : :         А:]Y
4
_output_shapes"
 :         А
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
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_1:Н И
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_0
Т
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11864

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_6_layer_call_and_return_conditional_losses_12154

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
├
Э
"__inference_internal_grad_fn_12553
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аb
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
subSubsub/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
addAddV2add/x:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :         Аa
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :         А\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :         Аg
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :         Аc
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         Аa
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :         Аb
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         А:         А: : :         А:]Y
4
_output_shapes"
 :         А
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
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_1:Н И
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_0
Я

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_11356

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ъ
K
/__inference_max_pooling3d_1_layer_call_fn_11925

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A                                             * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11123Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
█
Э
"__inference_internal_grad_fn_12607
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
:         ll@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:         ll@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:         ll@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:         ll@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:         ll@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:         ll@W
SquareSquaremul_biasadd*
T0*/
_output_shapes
:         ll@b
mul_4Mulresult_grads_0
Square:y:0*
T0*/
_output_shapes
:         ll@^
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@\
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*/
_output_shapes
:         ll@^
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
:         ll@Y
IdentityIdentity	mul_7:z:0*
T0*/
_output_shapes
:         ll@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ll@:         ll@: : :         ll@:XT
/
_output_shapes
:         ll@
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
:         ll@
(
_user_specified_nameresult_grads_1:И Г
&
 _has_manual_control_dependencies(
/
_output_shapes
:         ll@
(
_user_specified_nameresult_grads_0
╚
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_12040

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_5_layer_call_and_return_conditional_losses_12107

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_11486

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
э
p
F__inference_concatenate_layer_call_and_return_conditional_losses_11320

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:         А`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:XT
0
_output_shapes
:         А
 
_user_specified_nameinputs:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_12181

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ы
Х
'__inference_dense_3_layer_call_fn_12190

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_11426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12186:%!

_user_specified_name12184:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╪
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11930

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
╪
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11143

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
╧
b
)__inference_dropout_4_layer_call_fn_12159

inputs
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_11414p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_11497

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Я

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_12082

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11940

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я
Э
"__inference_internal_grad_fn_12391
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1m
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:         АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:         А^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         А[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:         А]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:         АX
SquareSquaremul_biasadd*
T0*0
_output_shapes
:         Аc
mul_4Mulresult_grads_0
Square:y:0*
T0*0
_output_shapes
:         А_
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*0
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         А]
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*0
_output_shapes
:         А^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: b
mul_7Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:         АZ
IdentityIdentity	mul_7:z:0*
T0*0
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         А:         А: : :         А:YU
0
_output_shapes
:         А
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
_user_specified_nameresult_grads_2:`\
0
_output_shapes
:         А
(
_user_specified_nameresult_grads_1:Й Д
&
 _has_manual_control_dependencies(
0
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
Я
▀
%__inference_model_layer_call_fn_11608
input_3
input_2%
unknown:@
	unknown_0:@#
	unknown_1:@
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А$
	unknown_5:@А
	unknown_6:	А)
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А

unknown_18:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCallinput_3input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11516o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11604:%!

_user_specified_name11602:%!

_user_specified_name11600:%!

_user_specified_name11598:%!

_user_specified_name11596:%!

_user_specified_name11594:%!

_user_specified_name11592:%!

_user_specified_name11590:%!

_user_specified_name11588:%!

_user_specified_name11586:%!

_user_specified_name11584:%
!

_user_specified_name11582:%	!

_user_specified_name11580:%!

_user_specified_name11578:%!

_user_specified_name11576:%!

_user_specified_name11574:%!

_user_specified_name11572:%!

_user_specified_name11570:%!

_user_specified_name11568:%!

_user_specified_name11566:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_2:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_3
▒
E
)__inference_flatten_1_layer_call_fn_12034

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_11327a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╠
╗
"__inference_internal_grad_fn_12742
result_grads_0
result_grads_1
result_grads_2
mul_model_conv3d_1_beta
mul_model_conv3d_1_biasadd
identity

identity_1П
mulMulmul_model_conv3d_1_betamul_model_conv3d_1_biasadd^result_grads_0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         АА
mul_1Mulmul_model_conv3d_1_betamul_model_conv3d_1_biasadd*
T0*4
_output_shapes"
 :         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
subSubsub/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
addAddV2add/x:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :         Аa
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :         Аk
SquareSquaremul_model_conv3d_1_biasadd*
T0*4
_output_shapes"
 :         Аg
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :         Аc
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         Аa
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :         Аb
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         А:         А: : :         А:lh
4
_output_shapes"
 :         А
0
_user_specified_namemodel/conv3d_1/BiasAdd:KG

_output_shapes
: 
-
_user_specified_namemodel/conv3d_1/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_1:Н И
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_0
я
Э
"__inference_internal_grad_fn_12364
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1m
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:         АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:         А^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         А[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:         А]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:         АX
SquareSquaremul_biasadd*
T0*0
_output_shapes
:         Аc
mul_4Mulresult_grads_0
Square:y:0*
T0*0
_output_shapes
:         А_
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*0
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:         А]
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*0
_output_shapes
:         А^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: b
mul_7Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:         АZ
IdentityIdentity	mul_7:z:0*
T0*0
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:         А:         А: : :         А:YU
0
_output_shapes
:         А
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
_user_specified_nameresult_grads_2:`\
0
_output_shapes
:         А
(
_user_specified_nameresult_grads_1:Й Д
&
 _has_manual_control_dependencies(
0
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
╛
Д
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11229

inputs=
conv3d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpБ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0а
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аj
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :         А╒
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11220*V
_output_shapesD
B:         А:         А: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :         АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         	@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:         	@
 
_user_specified_nameinputs
·
■
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11844

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:         ll@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:         ll@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:         ll@╦
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11835*L
_output_shapes:
8:         ll@:         ll@: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:         ll@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         pp: : 20
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
:         pp
 
_user_specified_nameinputs
Щ
а
(__inference_conv2d_5_layer_call_fn_11977

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11305x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         

А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11973:%!

_user_specified_name11971:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
Я

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_11385

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
З
А
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11920

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:           АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:           Аf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           АZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:           А═
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11911*N
_output_shapes<
::           А:           А: l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:           АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$@: : 20
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
:         $$@
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_4_layer_call_fn_11935

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11133Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
█
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_12087

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
я
Ч
'__inference_dense_4_layer_call_fn_12049

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_11339p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12045:%!

_user_specified_name12043:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▄
W
+__inference_concatenate_layer_call_fn_12022
inputs_0
inputs_1
identity╟
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_11320i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:ZV
0
_output_shapes
:         А
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:         А
"
_user_specified_name
inputs_0
·
■
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11205

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:         ll@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:         ll@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:         ll@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:         ll@╦
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11196*L
_output_shapes:
8:         ll@:         ll@: k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:         ll@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         pp: : 20
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
:         pp
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_6_layer_call_and_return_conditional_losses_11397

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
╛
Д
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11892

inputs=
conv3d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpБ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0а
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аj
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :         А╒
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11883*V
_output_shapesD
B:         А:         А: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :         АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         	@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:         	@
 
_user_specified_nameinputs
¤
▌
#__inference_signature_wrapper_11788
input_2
input_3%
unknown:@
	unknown_0:@#
	unknown_1:@
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А$
	unknown_5:@А
	unknown_6:	А)
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А

unknown_18:
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinput_3input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_11098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         
0@:         pp: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11784:%!

_user_specified_name11782:%!

_user_specified_name11780:%!

_user_specified_name11778:%!

_user_specified_name11776:%!

_user_specified_name11774:%!

_user_specified_name11772:%!

_user_specified_name11770:%!

_user_specified_name11768:%!

_user_specified_name11766:%!

_user_specified_name11764:%
!

_user_specified_name11762:%	!

_user_specified_name11760:%!

_user_specified_name11758:%!

_user_specified_name11756:%!

_user_specified_name11754:%!

_user_specified_name11752:%!

_user_specified_name11750:%!

_user_specified_name11748:%!

_user_specified_name11746:XT
/
_output_shapes
:         pp
!
_user_specified_name	input_3:\ X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_2
╒

Ў
B__inference_dense_4_layer_call_and_return_conditional_losses_12060

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_5_layer_call_fn_12011

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_11153Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
п
А
A__inference_conv3d_layer_call_and_return_conditional_losses_11816

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         	,<@*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         	,<@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
mulMulbeta:output:0BiasAdd:output:0*
T0*3
_output_shapes!
:         	,<@Y
SigmoidSigmoidmul:z:0*
T0*3
_output_shapes!
:         	,<@i
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@]
IdentityIdentity	mul_1:z:0*
T0*3
_output_shapes!
:         	,<@╙
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11807*T
_output_shapesB
@:         	,<@:         	,<@: o

Identity_1IdentityIdentityN:output:0^NoOp*
T0*3
_output_shapes!
:         	,<@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
0@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:         
0@
 
_user_specified_nameinputs
я
Э
"__inference_internal_grad_fn_12499
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1m
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:           АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:           А^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:           АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           А[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:           АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:           А]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:           АX
SquareSquaremul_biasadd*
T0*0
_output_shapes
:           Аc
mul_4Mulresult_grads_0
Square:y:0*
T0*0
_output_shapes
:           А_
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*0
_output_shapes
:           АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           А]
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*0
_output_shapes
:           А^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: b
mul_7Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:           АZ
IdentityIdentity	mul_7:z:0*
T0*0
_output_shapes
:           АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:           А:           А: : :           А:YU
0
_output_shapes
:           А
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
_user_specified_nameresult_grads_2:`\
0
_output_shapes
:           А
(
_user_specified_nameresult_grads_1:Й Д
&
 _has_manual_control_dependencies(
0
_output_shapes
:           А
(
_user_specified_nameresult_grads_0
├
Э
"__inference_internal_grad_fn_12445
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аb
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
subSubsub/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
addAddV2add/x:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :         Аa
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :         А\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :         Аg
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :         Аc
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         Аa
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :         Аb
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:         А:         А: : :         А:]Y
4
_output_shapes"
 :         А
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
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_1:Н И
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :         А
(
_user_specified_nameresult_grads_0
п
Э
"__inference_internal_grad_fn_12634
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1p
mulMulmul_betamul_biasadd^result_grads_0*
T0*3
_output_shapes!
:         	,<@Y
SigmoidSigmoidmul:z:0*
T0*3
_output_shapes!
:         	,<@a
mul_1Mulmul_betamul_biasadd*
T0*3
_output_shapes!
:         	,<@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
subSubsub/x:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@^
mul_2Mul	mul_1:z:0sub:z:0*
T0*3
_output_shapes!
:         	,<@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
addAddV2add/x:output:0	mul_2:z:0*
T0*3
_output_shapes!
:         	,<@`
mul_3MulSigmoid:y:0add:z:0*
T0*3
_output_shapes!
:         	,<@[
SquareSquaremul_biasadd*
T0*3
_output_shapes!
:         	,<@f
mul_4Mulresult_grads_0
Square:y:0*
T0*3
_output_shapes!
:         	,<@b
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@`
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*3
_output_shapes!
:         	,<@b
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: e
mul_7Mulresult_grads_0	mul_3:z:0*
T0*3
_output_shapes!
:         	,<@]
IdentityIdentity	mul_7:z:0*
T0*3
_output_shapes!
:         	,<@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         	,<@:         	,<@: : :         	,<@:\X
3
_output_shapes!
:         	,<@
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
_user_specified_nameresult_grads_2:c_
3
_output_shapes!
:         	,<@
(
_user_specified_nameresult_grads_1:М З
&
 _has_manual_control_dependencies(
3
_output_shapes!
:         	,<@
(
_user_specified_nameresult_grads_0
ў
╗
"__inference_internal_grad_fn_12769
result_grads_0
result_grads_1
result_grads_2
mul_model_conv2d_4_beta
mul_model_conv2d_4_biasadd
identity

identity_1Л
mulMulmul_model_conv2d_4_betamul_model_conv2d_4_biasadd^result_grads_0*
T0*0
_output_shapes
:           АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:           А|
mul_1Mulmul_model_conv2d_4_betamul_model_conv2d_4_biasadd*
T0*0
_output_shapes
:           АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           А[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:           АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:           А]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:           Аg
SquareSquaremul_model_conv2d_4_biasadd*
T0*0
_output_shapes
:           Аc
mul_4Mulresult_grads_0
Square:y:0*
T0*0
_output_shapes
:           А_
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*0
_output_shapes
:           АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:           А]
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*0
_output_shapes
:           А^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: b
mul_7Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:           АZ
IdentityIdentity	mul_7:z:0*
T0*0
_output_shapes
:           АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:           А:           А: : :           А:hd
0
_output_shapes
:           А
0
_user_specified_namemodel/conv2d_4/BiasAdd:KG

_output_shapes
: 
-
_user_specified_namemodel/conv2d_4/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:`\
0
_output_shapes
:           А
(
_user_specified_nameresult_grads_1:Й Д
&
 _has_manual_control_dependencies(
0
_output_shapes
:           А
(
_user_specified_nameresult_grads_0
╧
b
)__inference_dropout_3_layer_call_fn_12112

inputs
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_11385p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
н
д
(__inference_conv3d_2_layer_call_fn_11949

inputs'
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11279|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11945:%!

_user_specified_name11943:\ X
4
_output_shapes"
 :         А
 
_user_specified_nameinputs
е
╖
"__inference_internal_grad_fn_12688
result_grads_0
result_grads_1
result_grads_2
mul_model_conv3d_beta
mul_model_conv3d_biasadd
identity

identity_1К
mulMulmul_model_conv3d_betamul_model_conv3d_biasadd^result_grads_0*
T0*3
_output_shapes!
:         	,<@Y
SigmoidSigmoidmul:z:0*
T0*3
_output_shapes!
:         	,<@{
mul_1Mulmul_model_conv3d_betamul_model_conv3d_biasadd*
T0*3
_output_shapes!
:         	,<@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
subSubsub/x:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@^
mul_2Mul	mul_1:z:0sub:z:0*
T0*3
_output_shapes!
:         	,<@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
addAddV2add/x:output:0	mul_2:z:0*
T0*3
_output_shapes!
:         	,<@`
mul_3MulSigmoid:y:0add:z:0*
T0*3
_output_shapes!
:         	,<@h
SquareSquaremul_model_conv3d_biasadd*
T0*3
_output_shapes!
:         	,<@f
mul_4Mulresult_grads_0
Square:y:0*
T0*3
_output_shapes!
:         	,<@b
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@`
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*3
_output_shapes!
:         	,<@b
ConstConst*
_output_shapes
:*
dtype0*)
value B"                F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: e
mul_7Mulresult_grads_0	mul_3:z:0*
T0*3
_output_shapes!
:         	,<@]
IdentityIdentity	mul_7:z:0*
T0*3
_output_shapes!
:         	,<@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         	,<@:         	,<@: : :         	,<@:ie
3
_output_shapes!
:         	,<@
.
_user_specified_namemodel/conv3d/BiasAdd:IE

_output_shapes
: 
+
_user_specified_namemodel/conv3d/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:c_
3
_output_shapes!
:         	,<@
(
_user_specified_nameresult_grads_1:М З
&
 _has_manual_control_dependencies(
3
_output_shapes!
:         	,<@
(
_user_specified_nameresult_grads_0
┬
Е
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11279

inputs>
conv3d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpВ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:АА*
dtype0а
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :         АZ
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :         Аj
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :         А^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :         А╒
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11270*V
_output_shapesD
B:         А:         А: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :         АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :         А
 
_user_specified_nameinputs
╪
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11123

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_11153

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_3_layer_call_fn_11859

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11113Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_12016

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
█
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_12134

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ц
I
-__inference_max_pooling3d_layer_call_fn_11849

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A                                             * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11103Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A                                             "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
Ў
r
F__inference_concatenate_layer_call_and_return_conditional_losses_12029
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:         А`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:ZV
0
_output_shapes
:         А
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:         А
"
_user_specified_name
inputs_0
Т
Э
(__inference_conv2d_3_layer_call_fn_11825

inputs!
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11205w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ll@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11821:%!

_user_specified_name11819:W S
/
_output_shapes
:         pp
 
_user_specified_nameinputs
нн
▐)
!__inference__traced_restore_13313
file_prefix<
assignvariableop_conv3d_kernel:@,
assignvariableop_1_conv3d_bias:@<
"assignvariableop_2_conv2d_3_kernel:@.
 assignvariableop_3_conv2d_3_bias:@A
"assignvariableop_4_conv3d_1_kernel:@А/
 assignvariableop_5_conv3d_1_bias:	А=
"assignvariableop_6_conv2d_4_kernel:@А/
 assignvariableop_7_conv2d_4_bias:	АB
"assignvariableop_8_conv3d_2_kernel:АА/
 assignvariableop_9_conv3d_2_bias:	А?
#assignvariableop_10_conv2d_5_kernel:АА0
!assignvariableop_11_conv2d_5_bias:	А6
"assignvariableop_12_dense_4_kernel:
АА/
 assignvariableop_13_dense_4_bias:	А6
"assignvariableop_14_dense_5_kernel:
АА/
 assignvariableop_15_dense_5_bias:	А6
"assignvariableop_16_dense_6_kernel:
АА/
 assignvariableop_17_dense_6_bias:	А5
"assignvariableop_18_dense_3_kernel:	А.
 assignvariableop_19_dense_3_bias:'
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: F
(assignvariableop_22_adam_m_conv3d_kernel:@F
(assignvariableop_23_adam_v_conv3d_kernel:@4
&assignvariableop_24_adam_m_conv3d_bias:@4
&assignvariableop_25_adam_v_conv3d_bias:@D
*assignvariableop_26_adam_m_conv2d_3_kernel:@D
*assignvariableop_27_adam_v_conv2d_3_kernel:@6
(assignvariableop_28_adam_m_conv2d_3_bias:@6
(assignvariableop_29_adam_v_conv2d_3_bias:@I
*assignvariableop_30_adam_m_conv3d_1_kernel:@АI
*assignvariableop_31_adam_v_conv3d_1_kernel:@А7
(assignvariableop_32_adam_m_conv3d_1_bias:	А7
(assignvariableop_33_adam_v_conv3d_1_bias:	АE
*assignvariableop_34_adam_m_conv2d_4_kernel:@АE
*assignvariableop_35_adam_v_conv2d_4_kernel:@А7
(assignvariableop_36_adam_m_conv2d_4_bias:	А7
(assignvariableop_37_adam_v_conv2d_4_bias:	АJ
*assignvariableop_38_adam_m_conv3d_2_kernel:ААJ
*assignvariableop_39_adam_v_conv3d_2_kernel:АА7
(assignvariableop_40_adam_m_conv3d_2_bias:	А7
(assignvariableop_41_adam_v_conv3d_2_bias:	АF
*assignvariableop_42_adam_m_conv2d_5_kernel:ААF
*assignvariableop_43_adam_v_conv2d_5_kernel:АА7
(assignvariableop_44_adam_m_conv2d_5_bias:	А7
(assignvariableop_45_adam_v_conv2d_5_bias:	А=
)assignvariableop_46_adam_m_dense_4_kernel:
АА=
)assignvariableop_47_adam_v_dense_4_kernel:
АА6
'assignvariableop_48_adam_m_dense_4_bias:	А6
'assignvariableop_49_adam_v_dense_4_bias:	А=
)assignvariableop_50_adam_m_dense_5_kernel:
АА=
)assignvariableop_51_adam_v_dense_5_kernel:
АА6
'assignvariableop_52_adam_m_dense_5_bias:	А6
'assignvariableop_53_adam_v_dense_5_bias:	А=
)assignvariableop_54_adam_m_dense_6_kernel:
АА=
)assignvariableop_55_adam_v_dense_6_kernel:
АА6
'assignvariableop_56_adam_m_dense_6_bias:	А6
'assignvariableop_57_adam_v_dense_6_bias:	А<
)assignvariableop_58_adam_m_dense_3_kernel:	А<
)assignvariableop_59_adam_v_dense_3_kernel:	А5
'assignvariableop_60_adam_m_dense_3_bias:5
'assignvariableop_61_adam_v_dense_3_bias:%
assignvariableop_62_total_1: %
assignvariableop_63_count_1: #
assignvariableop_64_total: #
assignvariableop_65_count: 
identity_67ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9п
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*╒
value╦B╚CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH∙
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ё
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*в
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_4_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_4_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_4_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_4_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_5_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_5_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_6_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_6_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_20AssignVariableOpassignvariableop_20_iterationIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_m_conv3d_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_v_conv3d_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_m_conv3d_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_v_conv3d_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_conv2d_3_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_conv2d_3_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_conv2d_3_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_conv2d_3_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_conv3d_1_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_conv3d_1_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_conv3d_1_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_conv3d_1_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_conv2d_4_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_conv2d_4_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_conv2d_4_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_conv2d_4_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_conv3d_2_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_conv3d_2_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_conv3d_2_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_conv3d_2_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_conv2d_5_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_conv2d_5_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_conv2d_5_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_conv2d_5_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_m_dense_4_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_v_dense_4_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_m_dense_4_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_v_dense_4_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_m_dense_5_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_v_dense_5_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_m_dense_5_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_v_dense_5_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_m_dense_6_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_v_dense_6_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_m_dense_6_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_v_dense_6_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_m_dense_3_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_v_dense_3_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_m_dense_3_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_v_dense_3_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_62AssignVariableOpassignvariableop_62_total_1Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_63AssignVariableOpassignvariableop_63_count_1Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_64AssignVariableOpassignvariableop_64_totalIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_65AssignVariableOpassignvariableop_65_countIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 √
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_67IdentityIdentity_66:output:0^NoOp_1*
T0*
_output_shapes
: ─
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_67Identity_67:output:0*(
_construction_contextkEagerRuntime*Ы
_input_shapesЙ
Ж: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%B!

_user_specified_namecount:%A!

_user_specified_nametotal:'@#
!
_user_specified_name	count_1:'?#
!
_user_specified_name	total_1:3>/
-
_user_specified_nameAdam/v/dense_3/bias:3=/
-
_user_specified_nameAdam/m/dense_3/bias:5<1
/
_user_specified_nameAdam/v/dense_3/kernel:5;1
/
_user_specified_nameAdam/m/dense_3/kernel:3:/
-
_user_specified_nameAdam/v/dense_6/bias:39/
-
_user_specified_nameAdam/m/dense_6/bias:581
/
_user_specified_nameAdam/v/dense_6/kernel:571
/
_user_specified_nameAdam/m/dense_6/kernel:36/
-
_user_specified_nameAdam/v/dense_5/bias:35/
-
_user_specified_nameAdam/m/dense_5/bias:541
/
_user_specified_nameAdam/v/dense_5/kernel:531
/
_user_specified_nameAdam/m/dense_5/kernel:32/
-
_user_specified_nameAdam/v/dense_4/bias:31/
-
_user_specified_nameAdam/m/dense_4/bias:501
/
_user_specified_nameAdam/v/dense_4/kernel:5/1
/
_user_specified_nameAdam/m/dense_4/kernel:4.0
.
_user_specified_nameAdam/v/conv2d_5/bias:4-0
.
_user_specified_nameAdam/m/conv2d_5/bias:6,2
0
_user_specified_nameAdam/v/conv2d_5/kernel:6+2
0
_user_specified_nameAdam/m/conv2d_5/kernel:4*0
.
_user_specified_nameAdam/v/conv3d_2/bias:4)0
.
_user_specified_nameAdam/m/conv3d_2/bias:6(2
0
_user_specified_nameAdam/v/conv3d_2/kernel:6'2
0
_user_specified_nameAdam/m/conv3d_2/kernel:4&0
.
_user_specified_nameAdam/v/conv2d_4/bias:4%0
.
_user_specified_nameAdam/m/conv2d_4/bias:6$2
0
_user_specified_nameAdam/v/conv2d_4/kernel:6#2
0
_user_specified_nameAdam/m/conv2d_4/kernel:4"0
.
_user_specified_nameAdam/v/conv3d_1/bias:4!0
.
_user_specified_nameAdam/m/conv3d_1/bias:6 2
0
_user_specified_nameAdam/v/conv3d_1/kernel:62
0
_user_specified_nameAdam/m/conv3d_1/kernel:40
.
_user_specified_nameAdam/v/conv2d_3/bias:40
.
_user_specified_nameAdam/m/conv2d_3/bias:62
0
_user_specified_nameAdam/v/conv2d_3/kernel:62
0
_user_specified_nameAdam/m/conv2d_3/kernel:2.
,
_user_specified_nameAdam/v/conv3d/bias:2.
,
_user_specified_nameAdam/m/conv3d/bias:40
.
_user_specified_nameAdam/v/conv3d/kernel:40
.
_user_specified_nameAdam/m/conv3d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_4/bias:.*
(
_user_specified_namedense_4/kernel:-)
'
_user_specified_nameconv2d_5/bias:/+
)
_user_specified_nameconv2d_5/kernel:-
)
'
_user_specified_nameconv3d_2/bias:/	+
)
_user_specified_nameconv3d_2/kernel:-)
'
_user_specified_nameconv2d_4/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_3/kernel:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
▀в
С
 __inference__wrapped_model_11098
input_3
input_2I
+model_conv3d_conv3d_readvariableop_resource:@:
,model_conv3d_biasadd_readvariableop_resource:@G
-model_conv2d_3_conv2d_readvariableop_resource:@<
.model_conv2d_3_biasadd_readvariableop_resource:@L
-model_conv3d_1_conv3d_readvariableop_resource:@А=
.model_conv3d_1_biasadd_readvariableop_resource:	АH
-model_conv2d_4_conv2d_readvariableop_resource:@А=
.model_conv2d_4_biasadd_readvariableop_resource:	АM
-model_conv3d_2_conv3d_readvariableop_resource:АА=
.model_conv3d_2_biasadd_readvariableop_resource:	АI
-model_conv2d_5_conv2d_readvariableop_resource:АА=
.model_conv2d_5_biasadd_readvariableop_resource:	А@
,model_dense_4_matmul_readvariableop_resource:
АА<
-model_dense_4_biasadd_readvariableop_resource:	А@
,model_dense_5_matmul_readvariableop_resource:
АА<
-model_dense_5_biasadd_readvariableop_resource:	А@
,model_dense_6_matmul_readvariableop_resource:
АА<
-model_dense_6_biasadd_readvariableop_resource:	А?
,model_dense_3_matmul_readvariableop_resource:	А;
-model_dense_3_biasadd_readvariableop_resource:
identityИв%model/conv2d_3/BiasAdd/ReadVariableOpв$model/conv2d_3/Conv2D/ReadVariableOpв%model/conv2d_4/BiasAdd/ReadVariableOpв$model/conv2d_4/Conv2D/ReadVariableOpв%model/conv2d_5/BiasAdd/ReadVariableOpв$model/conv2d_5/Conv2D/ReadVariableOpв#model/conv3d/BiasAdd/ReadVariableOpв"model/conv3d/Conv3D/ReadVariableOpв%model/conv3d_1/BiasAdd/ReadVariableOpв$model/conv3d_1/Conv3D/ReadVariableOpв%model/conv3d_2/BiasAdd/ReadVariableOpв$model/conv3d_2/Conv3D/ReadVariableOpв$model/dense_3/BiasAdd/ReadVariableOpв#model/dense_3/MatMul/ReadVariableOpв$model/dense_4/BiasAdd/ReadVariableOpв#model/dense_4/MatMul/ReadVariableOpв$model/dense_5/BiasAdd/ReadVariableOpв#model/dense_5/MatMul/ReadVariableOpв$model/dense_6/BiasAdd/ReadVariableOpв#model/dense_6/MatMul/ReadVariableOpЪ
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0║
model/conv3d/Conv3DConv3Dinput_2*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         	,<@*
paddingVALID*
strides	
М
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0и
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         	,<@V
model/conv3d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
model/conv3d/mulMulmodel/conv3d/beta:output:0model/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:         	,<@s
model/conv3d/SigmoidSigmoidmodel/conv3d/mul:z:0*
T0*3
_output_shapes!
:         	,<@Р
model/conv3d/mul_1Mulmodel/conv3d/BiasAdd:output:0model/conv3d/Sigmoid:y:0*
T0*3
_output_shapes!
:         	,<@w
model/conv3d/IdentityIdentitymodel/conv3d/mul_1:z:0*
T0*3
_output_shapes!
:         	,<@З
model/conv3d/IdentityN	IdentityNmodel/conv3d/mul_1:z:0model/conv3d/BiasAdd:output:0model/conv3d/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-10971*T
_output_shapesB
@:         	,<@:         	,<@: ╟
model/max_pooling3d/MaxPool3D	MaxPool3Dmodel/conv3d/IdentityN:output:0*
T0*3
_output_shapes!
:         	@*
ksize	
*
paddingVALID*
strides	
Ъ
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╣
model/conv2d_3/Conv2DConv2Dinput_3,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@*
paddingVALID*
strides
Р
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0к
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@X
model/conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Т
model/conv2d_3/mulMulmodel/conv2d_3/beta:output:0model/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         ll@s
model/conv2d_3/SigmoidSigmoidmodel/conv2d_3/mul:z:0*
T0*/
_output_shapes
:         ll@Т
model/conv2d_3/mul_1Mulmodel/conv2d_3/BiasAdd:output:0model/conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:         ll@w
model/conv2d_3/IdentityIdentitymodel/conv2d_3/mul_1:z:0*
T0*/
_output_shapes
:         ll@З
model/conv2d_3/IdentityN	IdentityNmodel/conv2d_3/mul_1:z:0model/conv2d_3/BiasAdd:output:0model/conv2d_3/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-10987*L
_output_shapes:
8:         ll@:         ll@: Я
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0▐
model/conv3d_1/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         А*
paddingVALID*
strides	
С
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0п
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         АX
model/conv3d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
model/conv3d_1/mulMulmodel/conv3d_1/beta:output:0model/conv3d_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :         Аx
model/conv3d_1/SigmoidSigmoidmodel/conv3d_1/mul:z:0*
T0*4
_output_shapes"
 :         АЧ
model/conv3d_1/mul_1Mulmodel/conv3d_1/BiasAdd:output:0model/conv3d_1/Sigmoid:y:0*
T0*4
_output_shapes"
 :         А|
model/conv3d_1/IdentityIdentitymodel/conv3d_1/mul_1:z:0*
T0*4
_output_shapes"
 :         АС
model/conv3d_1/IdentityN	IdentityNmodel/conv3d_1/mul_1:z:0model/conv3d_1/BiasAdd:output:0model/conv3d_1/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11002*V
_output_shapesD
B:         А:         А: ╕
model/max_pooling2d_3/MaxPoolMaxPool!model/conv2d_3/IdentityN:output:0*/
_output_shapes
:         $$@*
ksize
*
paddingVALID*
strides
╠
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_1/IdentityN:output:0*
T0*4
_output_shapes"
 :         А*
ksize	
*
paddingVALID*
strides	
Ы
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0┘
model/conv2d_4/Conv2DConv2D&model/max_pooling2d_3/MaxPool:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
С
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0л
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           АX
model/conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?У
model/conv2d_4/mulMulmodel/conv2d_4/beta:output:0model/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:           Аt
model/conv2d_4/SigmoidSigmoidmodel/conv2d_4/mul:z:0*
T0*0
_output_shapes
:           АУ
model/conv2d_4/mul_1Mulmodel/conv2d_4/BiasAdd:output:0model/conv2d_4/Sigmoid:y:0*
T0*0
_output_shapes
:           Аx
model/conv2d_4/IdentityIdentitymodel/conv2d_4/mul_1:z:0*
T0*0
_output_shapes
:           АЙ
model/conv2d_4/IdentityN	IdentityNmodel/conv2d_4/mul_1:z:0model/conv2d_4/BiasAdd:output:0model/conv2d_4/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11019*N
_output_shapes<
::           А:           А: а
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*,
_output_shapes
:АА*
dtype0р
model/conv3d_2/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         А*
paddingVALID*
strides	
С
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0п
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         АX
model/conv3d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
model/conv3d_2/mulMulmodel/conv3d_2/beta:output:0model/conv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :         Аx
model/conv3d_2/SigmoidSigmoidmodel/conv3d_2/mul:z:0*
T0*4
_output_shapes"
 :         АЧ
model/conv3d_2/mul_1Mulmodel/conv3d_2/BiasAdd:output:0model/conv3d_2/Sigmoid:y:0*
T0*4
_output_shapes"
 :         А|
model/conv3d_2/IdentityIdentitymodel/conv3d_2/mul_1:z:0*
T0*4
_output_shapes"
 :         АС
model/conv3d_2/IdentityN	IdentityNmodel/conv3d_2/mul_1:z:0model/conv3d_2/BiasAdd:output:0model/conv3d_2/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11034*V
_output_shapesD
B:         А:         А: ╣
model/max_pooling2d_4/MaxPoolMaxPool!model/conv2d_4/IdentityN:output:0*0
_output_shapes
:         

А*
ksize
*
paddingVALID*
strides
╠
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_2/IdentityN:output:0*
T0*4
_output_shapes"
 :         А*
ksize	
*
paddingVALID*
strides	
Ь
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┘
model/conv2d_5/Conv2DConv2D&model/max_pooling2d_4/MaxPool:output:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
С
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0л
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АX
model/conv2d_5/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?У
model/conv2d_5/mulMulmodel/conv2d_5/beta:output:0model/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:         Аt
model/conv2d_5/SigmoidSigmoidmodel/conv2d_5/mul:z:0*
T0*0
_output_shapes
:         АУ
model/conv2d_5/mul_1Mulmodel/conv2d_5/BiasAdd:output:0model/conv2d_5/Sigmoid:y:0*
T0*0
_output_shapes
:         Аx
model/conv2d_5/IdentityIdentitymodel/conv2d_5/mul_1:z:0*
T0*0
_output_shapes
:         АЙ
model/conv2d_5/IdentityN	IdentityNmodel/conv2d_5/mul_1:z:0model/conv2d_5/BiasAdd:output:0model/conv2d_5/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-11051*N
_output_shapes<
::         А:         А: ╣
model/max_pooling2d_5/MaxPoolMaxPool!model/conv2d_5/IdentityN:output:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
w
model/tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ▒
model/tf.reshape/ReshapeReshape(model/max_pooling3d_2/MaxPool3D:output:0'model/tf.reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         А_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :█
model/concatenate/concatConcatV2&model/max_pooling2d_5/MaxPool:output:0!model/tf.reshape/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:         Аf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ш
model/flatten_1/ReshapeReshape!model/concatenate/concat:output:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:         АТ
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0а
model/dense_4/MatMulMatMul model/flatten_1/Reshape:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
model/dense_4/ReluRelumodel/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         Аy
model/dropout_2/IdentityIdentity model/dense_4/Relu:activations:0*
T0*(
_output_shapes
:         АТ
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0б
model/dense_5/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
model/dense_5/ReluRelumodel/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:         Аy
model/dropout_3/IdentityIdentity model/dense_5/Relu:activations:0*
T0*(
_output_shapes
:         АТ
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0б
model/dense_6/MatMulMatMul!model/dropout_3/Identity:output:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
model/dense_6/ReluRelumodel/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         Аy
model/dropout_4/IdentityIdentity model/dense_6/Relu:activations:0*
T0*(
_output_shapes
:         АС
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0а
model/dense_3/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         О
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model/dense_3/SoftmaxSoftmaxmodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         n
IdentityIdentitymodel/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp:($
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
resource:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_2:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_3
╚
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_11327

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_5_layer_call_and_return_conditional_losses_11368

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
б
E
)__inference_dropout_2_layer_call_fn_12070

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_11486a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╤`
ї	
@__inference_model_layer_call_and_return_conditional_losses_11433
input_3
input_2*
conv3d_11181:@
conv3d_11183:@(
conv2d_3_11206:@
conv2d_3_11208:@-
conv3d_1_11230:@А
conv3d_1_11232:	А)
conv2d_4_11256:@А
conv2d_4_11258:	А.
conv3d_2_11280:АА
conv3d_2_11282:	А*
conv2d_5_11306:АА
conv2d_5_11308:	А!
dense_4_11340:
АА
dense_4_11342:	А!
dense_5_11369:
АА
dense_5_11371:	А!
dense_6_11398:
АА
dense_6_11400:	А 
dense_3_11427:	А
dense_3_11429:
identityИв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallвdense_6/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallЄ
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv3d_11181conv3d_11183*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         	,<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_11180ю
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         	@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11103Ў
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_3_11206conv2d_3_11208*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11205Ъ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_11230conv3d_1_11232*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11229Ё
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11113ї
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11123Ш
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_11256conv2d_4_11258*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11255Ь
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_11280conv3d_2_11282*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11279ё
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11133ї
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11143Ш
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_5_11306conv2d_5_11308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11305ё
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_11153q
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             е
tf.reshape/ReshapeReshape(max_pooling3d_2/PartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         АЖ
concatenate/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0tf.reshape/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_11320╪
flatten_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_11327Ж
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_11340dense_4_11342*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_11339ь
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_11356О
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_5_11369dense_5_11371*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_11368Р
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_11385О
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_6_11398dense_6_11400*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_11397Р
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_11414Н
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_3_11427dense_3_11429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_11426w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ц
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:%!

_user_specified_name11429:%!

_user_specified_name11427:%!

_user_specified_name11400:%!

_user_specified_name11398:%!

_user_specified_name11371:%!

_user_specified_name11369:%!

_user_specified_name11342:%!

_user_specified_name11340:%!

_user_specified_name11308:%!

_user_specified_name11306:%!

_user_specified_name11282:%
!

_user_specified_name11280:%	!

_user_specified_name11258:%!

_user_specified_name11256:%!

_user_specified_name11232:%!

_user_specified_name11230:%!

_user_specified_name11208:%!

_user_specified_name11206:%!

_user_specified_name11183:%!

_user_specified_name11181:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_2:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_3
╒

Ў
B__inference_dense_4_layer_call_and_return_conditional_losses_11339

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
Ц
Я
(__inference_conv2d_4_layer_call_fn_11901

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11255x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:           А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name11897:%!

_user_specified_name11895:W S
/
_output_shapes
:         $$@
 
_user_specified_nameinputs:
"__inference_internal_grad_fn_12364CustomGradient-11987:
"__inference_internal_grad_fn_12391CustomGradient-11296:
"__inference_internal_grad_fn_12418CustomGradient-11959:
"__inference_internal_grad_fn_12445CustomGradient-11270:
"__inference_internal_grad_fn_12472CustomGradient-11911:
"__inference_internal_grad_fn_12499CustomGradient-11246:
"__inference_internal_grad_fn_12526CustomGradient-11883:
"__inference_internal_grad_fn_12553CustomGradient-11220:
"__inference_internal_grad_fn_12580CustomGradient-11835:
"__inference_internal_grad_fn_12607CustomGradient-11196:
"__inference_internal_grad_fn_12634CustomGradient-11807:
"__inference_internal_grad_fn_12661CustomGradient-11171:
"__inference_internal_grad_fn_12688CustomGradient-10971:
"__inference_internal_grad_fn_12715CustomGradient-10987:
"__inference_internal_grad_fn_12742CustomGradient-11002:
"__inference_internal_grad_fn_12769CustomGradient-11019:
"__inference_internal_grad_fn_12796CustomGradient-11034:
"__inference_internal_grad_fn_12823CustomGradient-11051"зL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*√
serving_defaultч
G
input_2<
serving_default_input_2:0         
0@
C
input_38
serving_default_input_3:0         pp;
dense_30
StatefulPartitionedCall:0         tensorflow/serving/predict:╗Е
н
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
 	optimizer
!
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
▌
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op"
_tf_keras_layer
▌
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias
 3_jit_compiled_convolution_op"
_tf_keras_layer
е
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
е
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op"
_tf_keras_layer
▌
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op"
_tf_keras_layer
е
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
е
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op"
_tf_keras_layer
▌
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias
 o_jit_compiled_convolution_op"
_tf_keras_layer
е
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
е
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
(
|	keras_api"
_tf_keras_layer
и
}	variables
~trainable_variables
regularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses
Пkernel
	Рbias"
_tf_keras_layer
├
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator"
_tf_keras_layer
├
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Юkernel
	Яbias"
_tf_keras_layer
├
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
ж_random_generator"
_tf_keras_layer
├
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
нkernel
	оbias"
_tf_keras_layer
├
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
│__call__
+┤&call_and_return_all_conditional_losses
╡_random_generator"
_tf_keras_layer
├
╢	variables
╖trainable_variables
╕regularization_losses
╣	keras_api
║__call__
+╗&call_and_return_all_conditional_losses
╝kernel
	╜bias"
_tf_keras_layer
╛
(0
)1
12
23
F4
G5
O6
P7
d8
e9
m10
n11
П12
Р13
Ю14
Я15
н16
о17
╝18
╜19"
trackable_list_wrapper
╛
(0
)1
12
23
F4
G5
O6
P7
d8
e9
m10
n11
П12
Р13
Ю14
Я15
н16
о17
╝18
╜19"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┴
├trace_0
─trace_12Ж
%__inference_model_layer_call_fn_11562
%__inference_model_layer_call_fn_11608╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0z─trace_1
ў
┼trace_0
╞trace_12╝
@__inference_model_layer_call_and_return_conditional_losses_11433
@__inference_model_layer_call_and_return_conditional_losses_11516╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0z╞trace_1
╘B╤
 __inference__wrapped_model_11098input_3input_2"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
╟
_variables
╚_iterations
╔_learning_rate
╩_index_dict
╦
_momentums
╠_velocities
═_update_step_xla"
experimentalOptimizer
-
╬serving_default"
signature_map
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╧non_trainable_variables
╨layers
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
т
╘trace_02├
&__inference_conv3d_layer_call_fn_11797Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╘trace_0
¤
╒trace_02▐
A__inference_conv3d_layer_call_and_return_conditional_losses_11816Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╒trace_0
+:)@2conv3d/kernel
:@2conv3d/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
ф
█trace_02┼
(__inference_conv2d_3_layer_call_fn_11825Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z█trace_0
 
▄trace_02р
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11844Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▄trace_0
):'@2conv2d_3/kernel
:@2conv2d_3/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
щ
тtrace_02╩
-__inference_max_pooling3d_layer_call_fn_11849Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0
Д
уtrace_02х
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11854Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zуtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ы
щtrace_02╠
/__inference_max_pooling2d_3_layer_call_fn_11859Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
Ж
ъtrace_02ч
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11864Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zъtrace_0
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ф
Ёtrace_02┼
(__inference_conv3d_1_layer_call_fn_11873Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
 
ёtrace_02р
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11892Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zёtrace_0
.:,@А2conv3d_1/kernel
:А2conv3d_1/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ф
ўtrace_02┼
(__inference_conv2d_4_layer_call_fn_11901Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
 
°trace_02р
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11920Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
*:(@А2conv2d_4/kernel
:А2conv2d_4/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ы
■trace_02╠
/__inference_max_pooling3d_1_layer_call_fn_11925Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
Ж
 trace_02ч
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11930Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
ы
Еtrace_02╠
/__inference_max_pooling2d_4_layer_call_fn_11935Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
Ж
Жtrace_02ч
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11940Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ф
Мtrace_02┼
(__inference_conv3d_2_layer_call_fn_11949Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
 
Нtrace_02р
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11968Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
/:-АА2conv3d_2/kernel
:А2conv3d_2/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
ф
Уtrace_02┼
(__inference_conv2d_5_layer_call_fn_11977Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
 
Фtrace_02р
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11996Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
+:)АА2conv2d_5/kernel
:А2conv2d_5/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
ы
Ъtrace_02╠
/__inference_max_pooling3d_2_layer_call_fn_12001Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
Ж
Ыtrace_02ч
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12006Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
ы
бtrace_02╠
/__inference_max_pooling2d_5_layer_call_fn_12011Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
Ж
вtrace_02ч
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_12016Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
}	variables
~trainable_variables
regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
ч
иtrace_02╚
+__inference_concatenate_layer_call_fn_12022Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0
В
йtrace_02у
F__inference_concatenate_layer_call_and_return_conditional_losses_12029Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
х
пtrace_02╞
)__inference_flatten_1_layer_call_fn_12034Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0
А
░trace_02с
D__inference_flatten_1_layer_call_and_return_conditional_losses_12040Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z░trace_0
0
П0
Р1"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
у
╢trace_02─
'__inference_dense_4_layer_call_fn_12049Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╢trace_0
■
╖trace_02▀
B__inference_dense_4_layer_call_and_return_conditional_losses_12060Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╖trace_0
": 
АА2dense_4/kernel
:А2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
╜
╜trace_0
╛trace_12В
)__inference_dropout_2_layer_call_fn_12065
)__inference_dropout_2_layer_call_fn_12070й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0z╛trace_1
є
┐trace_0
└trace_12╕
D__inference_dropout_2_layer_call_and_return_conditional_losses_12082
D__inference_dropout_2_layer_call_and_return_conditional_losses_12087й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┐trace_0z└trace_1
"
_generic_user_object
0
Ю0
Я1"
trackable_list_wrapper
0
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
у
╞trace_02─
'__inference_dense_5_layer_call_fn_12096Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╞trace_0
■
╟trace_02▀
B__inference_dense_5_layer_call_and_return_conditional_losses_12107Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╟trace_0
": 
АА2dense_5/kernel
:А2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
╜
═trace_0
╬trace_12В
)__inference_dropout_3_layer_call_fn_12112
)__inference_dropout_3_layer_call_fn_12117й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z═trace_0z╬trace_1
є
╧trace_0
╨trace_12╕
D__inference_dropout_3_layer_call_and_return_conditional_losses_12129
D__inference_dropout_3_layer_call_and_return_conditional_losses_12134й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╧trace_0z╨trace_1
"
_generic_user_object
0
н0
о1"
trackable_list_wrapper
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
у
╓trace_02─
'__inference_dense_6_layer_call_fn_12143Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╓trace_0
■
╫trace_02▀
B__inference_dense_6_layer_call_and_return_conditional_losses_12154Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0
": 
АА2dense_6/kernel
:А2dense_6/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╪non_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
п	variables
░trainable_variables
▒regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
╜
▌trace_0
▐trace_12В
)__inference_dropout_4_layer_call_fn_12159
)__inference_dropout_4_layer_call_fn_12164й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▌trace_0z▐trace_1
є
▀trace_0
рtrace_12╕
D__inference_dropout_4_layer_call_and_return_conditional_losses_12176
D__inference_dropout_4_layer_call_and_return_conditional_losses_12181й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0zрtrace_1
"
_generic_user_object
0
╝0
╜1"
trackable_list_wrapper
0
╝0
╜1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
╢	variables
╖trainable_variables
╕regularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
у
цtrace_02─
'__inference_dense_3_layer_call_fn_12190Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0
■
чtrace_02▀
B__inference_dense_3_layer_call_and_return_conditional_losses_12201Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
!:	А2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
╓
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
23"
trackable_list_wrapper
0
ш0
щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
%__inference_model_layer_call_fn_11562input_3input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
%__inference_model_layer_call_fn_11608input_3input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
@__inference_model_layer_call_and_return_conditional_losses_11433input_3input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
@__inference_model_layer_call_and_return_conditional_losses_11516input_3input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
З
╚0
ъ1
ы2
ь3
э4
ю5
я6
Ё7
ё8
Є9
є10
Ї11
ї12
Ў13
ў14
°15
∙16
·17
√18
№19
¤20
■21
 22
А23
Б24
В25
Г26
Д27
Е28
Ж29
З30
И31
Й32
К33
Л34
М35
Н36
О37
П38
Р39
С40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
╩
ъ0
ь1
ю2
Ё3
Є4
Ї5
Ў6
°7
·8
№9
■10
А11
В12
Д13
Ж14
И15
К16
М17
О18
Р19"
trackable_list_wrapper
╩
ы0
э1
я2
ё3
є4
ї5
ў6
∙7
√8
¤9
 10
Б11
Г12
Е13
З14
Й15
Л16
Н17
П18
С19"
trackable_list_wrapper
╡2▓п
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
сB▐
#__inference_signature_wrapper_11788input_2input_3"д
Э▓Щ
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 '

kwonlyargsЪ
	jinput_2
	jinput_3
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_conv3d_layer_call_fn_11797inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_conv3d_layer_call_and_return_conditional_losses_11816inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╥B╧
(__inference_conv2d_3_layer_call_fn_11825inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11844inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╫B╘
-__inference_max_pooling3d_layer_call_fn_11849inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11854inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┘B╓
/__inference_max_pooling2d_3_layer_call_fn_11859inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11864inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╥B╧
(__inference_conv3d_1_layer_call_fn_11873inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11892inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╥B╧
(__inference_conv2d_4_layer_call_fn_11901inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11920inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┘B╓
/__inference_max_pooling3d_1_layer_call_fn_11925inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11930inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┘B╓
/__inference_max_pooling2d_4_layer_call_fn_11935inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11940inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╥B╧
(__inference_conv3d_2_layer_call_fn_11949inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11968inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╥B╧
(__inference_conv2d_5_layer_call_fn_11977inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11996inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┘B╓
/__inference_max_pooling3d_2_layer_call_fn_12001inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12006inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┘B╓
/__inference_max_pooling2d_5_layer_call_fn_12011inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_12016inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
сB▐
+__inference_concatenate_layer_call_fn_12022inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_concatenate_layer_call_and_return_conditional_losses_12029inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╙B╨
)__inference_flatten_1_layer_call_fn_12034inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_flatten_1_layer_call_and_return_conditional_losses_12040inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_dense_4_layer_call_fn_12049inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_4_layer_call_and_return_conditional_losses_12060inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
)__inference_dropout_2_layer_call_fn_12065inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀B▄
)__inference_dropout_2_layer_call_fn_12070inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_2_layer_call_and_return_conditional_losses_12082inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_2_layer_call_and_return_conditional_losses_12087inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_dense_5_layer_call_fn_12096inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_5_layer_call_and_return_conditional_losses_12107inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
)__inference_dropout_3_layer_call_fn_12112inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀B▄
)__inference_dropout_3_layer_call_fn_12117inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_3_layer_call_and_return_conditional_losses_12129inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_3_layer_call_and_return_conditional_losses_12134inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_dense_6_layer_call_fn_12143inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_6_layer_call_and_return_conditional_losses_12154inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
)__inference_dropout_4_layer_call_fn_12159inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀B▄
)__inference_dropout_4_layer_call_fn_12164inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_4_layer_call_and_return_conditional_losses_12176inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_4_layer_call_and_return_conditional_losses_12181inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_dense_3_layer_call_fn_12190inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_3_layer_call_and_return_conditional_losses_12201inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
Т	variables
У	keras_api

Фtotal

Хcount"
_tf_keras_metric
c
Ц	variables
Ч	keras_api

Шtotal

Щcount
Ъ
_fn_kwargs"
_tf_keras_metric
0:.@2Adam/m/conv3d/kernel
0:.@2Adam/v/conv3d/kernel
:@2Adam/m/conv3d/bias
:@2Adam/v/conv3d/bias
.:,@2Adam/m/conv2d_3/kernel
.:,@2Adam/v/conv2d_3/kernel
 :@2Adam/m/conv2d_3/bias
 :@2Adam/v/conv2d_3/bias
3:1@А2Adam/m/conv3d_1/kernel
3:1@А2Adam/v/conv3d_1/kernel
!:А2Adam/m/conv3d_1/bias
!:А2Adam/v/conv3d_1/bias
/:-@А2Adam/m/conv2d_4/kernel
/:-@А2Adam/v/conv2d_4/kernel
!:А2Adam/m/conv2d_4/bias
!:А2Adam/v/conv2d_4/bias
4:2АА2Adam/m/conv3d_2/kernel
4:2АА2Adam/v/conv3d_2/kernel
!:А2Adam/m/conv3d_2/bias
!:А2Adam/v/conv3d_2/bias
0:.АА2Adam/m/conv2d_5/kernel
0:.АА2Adam/v/conv2d_5/kernel
!:А2Adam/m/conv2d_5/bias
!:А2Adam/v/conv2d_5/bias
':%
АА2Adam/m/dense_4/kernel
':%
АА2Adam/v/dense_4/kernel
 :А2Adam/m/dense_4/bias
 :А2Adam/v/dense_4/bias
':%
АА2Adam/m/dense_5/kernel
':%
АА2Adam/v/dense_5/kernel
 :А2Adam/m/dense_5/bias
 :А2Adam/v/dense_5/bias
':%
АА2Adam/m/dense_6/kernel
':%
АА2Adam/v/dense_6/kernel
 :А2Adam/m/dense_6/bias
 :А2Adam/v/dense_6/bias
&:$	А2Adam/m/dense_3/kernel
&:$	А2Adam/v/dense_3/kernel
:2Adam/m/dense_3/bias
:2Adam/v/dense_3/bias
0
Ф0
Х1"
trackable_list_wrapper
.
Т	variables"
_generic_user_object
:  (2total
:  (2count
0
Ш0
Щ1"
trackable_list_wrapper
.
Ц	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
ObM
beta:0C__inference_conv2d_5_layer_call_and_return_conditional_losses_11996
RbP
	BiasAdd:0C__inference_conv2d_5_layer_call_and_return_conditional_losses_11996
ObM
beta:0C__inference_conv2d_5_layer_call_and_return_conditional_losses_11305
RbP
	BiasAdd:0C__inference_conv2d_5_layer_call_and_return_conditional_losses_11305
ObM
beta:0C__inference_conv3d_2_layer_call_and_return_conditional_losses_11968
RbP
	BiasAdd:0C__inference_conv3d_2_layer_call_and_return_conditional_losses_11968
ObM
beta:0C__inference_conv3d_2_layer_call_and_return_conditional_losses_11279
RbP
	BiasAdd:0C__inference_conv3d_2_layer_call_and_return_conditional_losses_11279
ObM
beta:0C__inference_conv2d_4_layer_call_and_return_conditional_losses_11920
RbP
	BiasAdd:0C__inference_conv2d_4_layer_call_and_return_conditional_losses_11920
ObM
beta:0C__inference_conv2d_4_layer_call_and_return_conditional_losses_11255
RbP
	BiasAdd:0C__inference_conv2d_4_layer_call_and_return_conditional_losses_11255
ObM
beta:0C__inference_conv3d_1_layer_call_and_return_conditional_losses_11892
RbP
	BiasAdd:0C__inference_conv3d_1_layer_call_and_return_conditional_losses_11892
ObM
beta:0C__inference_conv3d_1_layer_call_and_return_conditional_losses_11229
RbP
	BiasAdd:0C__inference_conv3d_1_layer_call_and_return_conditional_losses_11229
ObM
beta:0C__inference_conv2d_3_layer_call_and_return_conditional_losses_11844
RbP
	BiasAdd:0C__inference_conv2d_3_layer_call_and_return_conditional_losses_11844
ObM
beta:0C__inference_conv2d_3_layer_call_and_return_conditional_losses_11205
RbP
	BiasAdd:0C__inference_conv2d_3_layer_call_and_return_conditional_losses_11205
MbK
beta:0A__inference_conv3d_layer_call_and_return_conditional_losses_11816
PbN
	BiasAdd:0A__inference_conv3d_layer_call_and_return_conditional_losses_11816
MbK
beta:0A__inference_conv3d_layer_call_and_return_conditional_losses_11180
PbN
	BiasAdd:0A__inference_conv3d_layer_call_and_return_conditional_losses_11180
9b7
model/conv3d/beta:0 __inference__wrapped_model_11098
<b:
model/conv3d/BiasAdd:0 __inference__wrapped_model_11098
;b9
model/conv2d_3/beta:0 __inference__wrapped_model_11098
>b<
model/conv2d_3/BiasAdd:0 __inference__wrapped_model_11098
;b9
model/conv3d_1/beta:0 __inference__wrapped_model_11098
>b<
model/conv3d_1/BiasAdd:0 __inference__wrapped_model_11098
;b9
model/conv2d_4/beta:0 __inference__wrapped_model_11098
>b<
model/conv2d_4/BiasAdd:0 __inference__wrapped_model_11098
;b9
model/conv3d_2/beta:0 __inference__wrapped_model_11098
>b<
model/conv3d_2/BiasAdd:0 __inference__wrapped_model_11098
;b9
model/conv2d_5/beta:0 __inference__wrapped_model_11098
>b<
model/conv2d_5/BiasAdd:0 __inference__wrapped_model_11098ф
 __inference__wrapped_model_11098┐()12FGOPdemnПРЮЯно╝╜lвi
bв_
]ЪZ
)К&
input_3         pp
-К*
input_2         
0@
к "1к.
,
dense_3!К
dense_3         Ё
F__inference_concatenate_layer_call_and_return_conditional_losses_12029еlвi
bв_
]ЪZ
+К(
inputs_0         А
+К(
inputs_1         А
к "5в2
+К(
tensor_0         А
Ъ ╩
+__inference_concatenate_layer_call_fn_12022Ъlвi
bв_
]ЪZ
+К(
inputs_0         А
+К(
inputs_1         А
к "*К'
unknown         А║
C__inference_conv2d_3_layer_call_and_return_conditional_losses_11844s127в4
-в*
(К%
inputs         pp
к "4в1
*К'
tensor_0         ll@
Ъ Ф
(__inference_conv2d_3_layer_call_fn_11825h127в4
-в*
(К%
inputs         pp
к ")К&
unknown         ll@╗
C__inference_conv2d_4_layer_call_and_return_conditional_losses_11920tOP7в4
-в*
(К%
inputs         $$@
к "5в2
+К(
tensor_0           А
Ъ Х
(__inference_conv2d_4_layer_call_fn_11901iOP7в4
-в*
(К%
inputs         $$@
к "*К'
unknown           А╝
C__inference_conv2d_5_layer_call_and_return_conditional_losses_11996umn8в5
.в+
)К&
inputs         

А
к "5в2
+К(
tensor_0         А
Ъ Ц
(__inference_conv2d_5_layer_call_fn_11977jmn8в5
.в+
)К&
inputs         

А
к "*К'
unknown         А├
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11892|FG;в8
1в.
,К)
inputs         	@
к "9в6
/К,
tensor_0         А
Ъ Э
(__inference_conv3d_1_layer_call_fn_11873qFG;в8
1в.
,К)
inputs         	@
к ".К+
unknown         А─
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11968}de<в9
2в/
-К*
inputs         А
к "9в6
/К,
tensor_0         А
Ъ Ю
(__inference_conv3d_2_layer_call_fn_11949rde<в9
2в/
-К*
inputs         А
к ".К+
unknown         А└
A__inference_conv3d_layer_call_and_return_conditional_losses_11816{();в8
1в.
,К)
inputs         
0@
к "8в5
.К+
tensor_0         	,<@
Ъ Ъ
&__inference_conv3d_layer_call_fn_11797p();в8
1в.
,К)
inputs         
0@
к "-К*
unknown         	,<@м
B__inference_dense_3_layer_call_and_return_conditional_losses_12201f╝╜0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Ж
'__inference_dense_3_layer_call_fn_12190[╝╜0в-
&в#
!К
inputs         А
к "!К
unknown         н
B__inference_dense_4_layer_call_and_return_conditional_losses_12060gПР0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
'__inference_dense_4_layer_call_fn_12049\ПР0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
B__inference_dense_5_layer_call_and_return_conditional_losses_12107gЮЯ0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
'__inference_dense_5_layer_call_fn_12096\ЮЯ0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
B__inference_dense_6_layer_call_and_return_conditional_losses_12154gно0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
'__inference_dense_6_layer_call_fn_12143\но0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
D__inference_dropout_2_layer_call_and_return_conditional_losses_12082e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_2_layer_call_and_return_conditional_losses_12087e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_2_layer_call_fn_12065Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_2_layer_call_fn_12070Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ан
D__inference_dropout_3_layer_call_and_return_conditional_losses_12129e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_3_layer_call_and_return_conditional_losses_12134e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_3_layer_call_fn_12112Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_3_layer_call_fn_12117Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ан
D__inference_dropout_4_layer_call_and_return_conditional_losses_12176e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_4_layer_call_and_return_conditional_losses_12181e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_4_layer_call_fn_12159Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_4_layer_call_fn_12164Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         А▒
D__inference_flatten_1_layer_call_and_return_conditional_losses_12040i8в5
.в+
)К&
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Л
)__inference_flatten_1_layer_call_fn_12034^8в5
.в+
)К&
inputs         А
к ""К
unknown         АМ
"__inference_internal_grad_fn_12364хЫЬУвП
ЗвГ

 
1К.
result_grads_0         А
1К.
result_grads_1         А
К
result_grads_2 
к "GЪD

 
+К(
tensor_1         А
К
tensor_2 М
"__inference_internal_grad_fn_12391хЭЮУвП
ЗвГ

 
1К.
result_grads_0         А
1К.
result_grads_1         А
К
result_grads_2 
к "GЪD

 
+К(
tensor_1         А
К
tensor_2 Ш
"__inference_internal_grad_fn_12418ёЯаЫвЧ
ПвЛ

 
5К2
result_grads_0         А
5К2
result_grads_1         А
К
result_grads_2 
к "KЪH

 
/К,
tensor_1         А
К
tensor_2 Ш
"__inference_internal_grad_fn_12445ёбвЫвЧ
ПвЛ

 
5К2
result_grads_0         А
5К2
result_grads_1         А
К
result_grads_2 
к "KЪH

 
/К,
tensor_1         А
К
tensor_2 М
"__inference_internal_grad_fn_12472хгдУвП
ЗвГ

 
1К.
result_grads_0           А
1К.
result_grads_1           А
К
result_grads_2 
к "GЪD

 
+К(
tensor_1           А
К
tensor_2 М
"__inference_internal_grad_fn_12499хежУвП
ЗвГ

 
1К.
result_grads_0           А
1К.
result_grads_1           А
К
result_grads_2 
к "GЪD

 
+К(
tensor_1           А
К
tensor_2 Ш
"__inference_internal_grad_fn_12526ёзиЫвЧ
ПвЛ

 
5К2
result_grads_0         А
5К2
result_grads_1         А
К
result_grads_2 
к "KЪH

 
/К,
tensor_1         А
К
tensor_2 Ш
"__inference_internal_grad_fn_12553ёйкЫвЧ
ПвЛ

 
5К2
result_grads_0         А
5К2
result_grads_1         А
К
result_grads_2 
к "KЪH

 
/К,
tensor_1         А
К
tensor_2 Й
"__inference_internal_grad_fn_12580тлмСвН
ЕвБ

 
0К-
result_grads_0         ll@
0К-
result_grads_1         ll@
К
result_grads_2 
к "FЪC

 
*К'
tensor_1         ll@
К
tensor_2 Й
"__inference_internal_grad_fn_12607тноСвН
ЕвБ

 
0К-
result_grads_0         ll@
0К-
result_grads_1         ll@
К
result_grads_2 
к "FЪC

 
*К'
tensor_1         ll@
К
tensor_2 Х
"__inference_internal_grad_fn_12634юп░ЩвХ
НвЙ

 
4К1
result_grads_0         	,<@
4К1
result_grads_1         	,<@
К
result_grads_2 
к "JЪG

 
.К+
tensor_1         	,<@
К
tensor_2 Х
"__inference_internal_grad_fn_12661ю▒▓ЩвХ
НвЙ

 
4К1
result_grads_0         	,<@
4К1
result_grads_1         	,<@
К
result_grads_2 
к "JЪG

 
.К+
tensor_1         	,<@
К
tensor_2 Х
"__inference_internal_grad_fn_12688ю│┤ЩвХ
НвЙ

 
4К1
result_grads_0         	,<@
4К1
result_grads_1         	,<@
К
result_grads_2 
к "JЪG

 
.К+
tensor_1         	,<@
К
tensor_2 Й
"__inference_internal_grad_fn_12715т╡╢СвН
ЕвБ

 
0К-
result_grads_0         ll@
0К-
result_grads_1         ll@
К
result_grads_2 
к "FЪC

 
*К'
tensor_1         ll@
К
tensor_2 Ш
"__inference_internal_grad_fn_12742ё╖╕ЫвЧ
ПвЛ

 
5К2
result_grads_0         А
5К2
result_grads_1         А
К
result_grads_2 
к "KЪH

 
/К,
tensor_1         А
К
tensor_2 М
"__inference_internal_grad_fn_12769х╣║УвП
ЗвГ

 
1К.
result_grads_0           А
1К.
result_grads_1           А
К
result_grads_2 
к "GЪD

 
+К(
tensor_1           А
К
tensor_2 Ш
"__inference_internal_grad_fn_12796ё╗╝ЫвЧ
ПвЛ

 
5К2
result_grads_0         А
5К2
result_grads_1         А
К
result_grads_2 
к "KЪH

 
/К,
tensor_1         А
К
tensor_2 М
"__inference_internal_grad_fn_12823х╜╛УвП
ЗвГ

 
1К.
result_grads_0         А
1К.
result_grads_1         А
К
result_grads_2 
к "GЪD

 
+К(
tensor_1         А
К
tensor_2 Ї
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_11864еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_3_layer_call_fn_11859ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_11940еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_4_layer_call_fn_11935ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_12016еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_5_layer_call_fn_12011ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    О
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11930┐_в\
UвR
PКM
inputsA                                             
к "\вY
RКO
tensor_0A                                             
Ъ ш
/__inference_max_pooling3d_1_layer_call_fn_11925┤_в\
UвR
PКM
inputsA                                             
к "QКN
unknownA                                             О
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12006┐_в\
UвR
PКM
inputsA                                             
к "\вY
RКO
tensor_0A                                             
Ъ ш
/__inference_max_pooling3d_2_layer_call_fn_12001┤_в\
UвR
PКM
inputsA                                             
к "QКN
unknownA                                             М
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11854┐_в\
UвR
PКM
inputsA                                             
к "\вY
RКO
tensor_0A                                             
Ъ ц
-__inference_max_pooling3d_layer_call_fn_11849┤_в\
UвR
PКM
inputsA                                             
к "QКN
unknownA                                             З
@__inference_model_layer_call_and_return_conditional_losses_11433┬()12FGOPdemnПРЮЯно╝╜tвq
jвg
]ЪZ
)К&
input_3         pp
-К*
input_2         
0@
p

 
к ",в)
"К
tensor_0         
Ъ З
@__inference_model_layer_call_and_return_conditional_losses_11516┬()12FGOPdemnПРЮЯно╝╜tвq
jвg
]ЪZ
)К&
input_3         pp
-К*
input_2         
0@
p 

 
к ",в)
"К
tensor_0         
Ъ с
%__inference_model_layer_call_fn_11562╖()12FGOPdemnПРЮЯно╝╜tвq
jвg
]ЪZ
)К&
input_3         pp
-К*
input_2         
0@
p

 
к "!К
unknown         с
%__inference_model_layer_call_fn_11608╖()12FGOPdemnПРЮЯно╝╜tвq
jвg
]ЪZ
)К&
input_3         pp
-К*
input_2         
0@
p 

 
к "!К
unknown         °
#__inference_signature_wrapper_11788╨()12FGOPdemnПРЮЯно╝╜}вz
в 
sкp
8
input_2-К*
input_2         
0@
4
input_3)К&
input_3         pp"1к.
,
dense_3!К
dense_3         