║╠
с░
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
 И"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8иё
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
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:*
dtype0
Г
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	А*
dtype0
Г
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	А*
dtype0

Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_3/bias
x
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_3/bias
x
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_3/kernel
Б
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_3/kernel
Б
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel* 
_output_shapes
:
АА*
dtype0

Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_2/bias
x
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_2/bias
x
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_2/kernel
Б
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_2/kernel
Б
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel* 
_output_shapes
:
АА*
dtype0

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_1/kernel
Б
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_1/kernel
Б
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel* 
_output_shapes
:
АА*
dtype0
А
Adam/v/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_2/bias
y
(Adam/v/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_2/bias
y
(Adam/m/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/v/conv2d_2/kernel
Й
*Adam/v/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
Р
Adam/m/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/m/conv2d_2/kernel
Й
*Adam/m/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
А
Adam/v/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv3d_2/bias
y
(Adam/v/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv3d_2/bias
y
(Adam/m/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/bias*
_output_shapes
: *
dtype0
Ф
Adam/v/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/v/conv3d_2/kernel
Н
*Adam/v/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/kernel**
_output_shapes
:  *
dtype0
Ф
Adam/m/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/m/conv3d_2/kernel
Н
*Adam/m/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/kernel**
_output_shapes
:  *
dtype0
А
Adam/v/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_1/bias
y
(Adam/v/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_1/bias
y
(Adam/m/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/bias*
_output_shapes
: *
dtype0
Р
Adam/v/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_1/kernel
Й
*Adam/v/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/kernel*&
_output_shapes
: *
dtype0
Р
Adam/m/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_1/kernel
Й
*Adam/m/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/kernel*&
_output_shapes
: *
dtype0
А
Adam/v/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv3d_1/bias
y
(Adam/v/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv3d_1/bias
y
(Adam/m/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/bias*
_output_shapes
: *
dtype0
Ф
Adam/v/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv3d_1/kernel
Н
*Adam/v/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/kernel**
_output_shapes
: *
dtype0
Ф
Adam/m/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv3d_1/kernel
Н
*Adam/m/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/kernel**
_output_shapes
: *
dtype0
|
Adam/v/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv2d/bias
u
&Adam/v/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv2d/bias
u
&Adam/m/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d/kernel
Е
(Adam/v/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/kernel*&
_output_shapes
:*
dtype0
М
Adam/m/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d/kernel
Е
(Adam/m/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/kernel*&
_output_shapes
:*
dtype0
|
Adam/v/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv3d/bias
u
&Adam/v/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv3d/bias
u
&Adam/m/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv3d/kernel
Й
(Adam/v/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv3d/kernel
Й
(Adam/m/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/kernel**
_output_shapes
:*
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
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	А*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:А*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:А*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
АА*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
: *
dtype0
Ж
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:  *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
: *
dtype0
Ж
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
В
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0
Т
serving_default_input_1Placeholder*3
_output_shapes!
:         
0@*
dtype0*(
shape:         
0@
К
serving_default_input_2Placeholder*/
_output_shapes
:         pp*
dtype0*$
shape:         pp
Я
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv3d/kernelconv3d/biasconv2d/kernelconv2d/biasconv3d_1/kernelconv3d_1/biasconv2d_1/kernelconv2d_1/biasconv3d_2/kernelconv3d_2/biasconv2d_2/kernelconv2d_2/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense/kernel
dense/bias*!
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
#__inference_signature_wrapper_12041

NoOpNoOp
┤г
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*юв
valueувB▀в B╫в
Ц
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-6
layer-18
layer-19
layer_with_weights-7
layer-20
layer-21
layer_with_weights-8
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
О
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
╚
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias
 0_jit_compiled_convolution_op*
╚
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op*
О
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
О
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
╚
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op*
╚
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op*
О
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
О
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
╚
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op*
╚
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op*
О
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
Р
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses* 

В	keras_api* 
Ф
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses* 
Ф
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses* 
о
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses
Хkernel
	Цbias*
м
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
Э_random_generator* 
о
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses
дkernel
	еbias*
м
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses
м_random_generator* 
о
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses
│kernel
	┤bias*
о
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗kernel
	╝bias*
в
.0
/1
72
83
L4
M5
U6
V7
j8
k9
s10
t11
Х12
Ц13
д14
е15
│16
┤17
╗18
╝19*
в
.0
/1
72
83
L4
M5
U6
V7
j8
k9
s10
t11
Х12
Ц13
д14
е15
│16
┤17
╗18
╝19*
* 
╡
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

┬trace_0
├trace_1* 

─trace_0
┼trace_1* 
* 
И
╞
_variables
╟_iterations
╚_learning_rate
╔_index_dict
╩
_momentums
╦_velocities
╠_update_step_xla*

═serving_default* 
* 
* 
* 
Ц
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 

╙trace_0* 

╘trace_0* 

.0
/1*

.0
/1*
* 
Ш
╒non_trainable_variables
╓layers
╫metrics
 ╪layer_regularization_losses
┘layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

┌trace_0* 

█trace_0* 
]W
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

70
81*

70
81*
* 
Ш
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

сtrace_0* 

тtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

шtrace_0* 

щtrace_0* 
* 
* 
* 
Ц
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

яtrace_0* 

Ёtrace_0* 

L0
M1*

L0
M1*
* 
Ш
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

Ўtrace_0* 

ўtrace_0* 
_Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

U0
V1*

U0
V1*
* 
Ш
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

¤trace_0* 

■trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

Дtrace_0* 

Еtrace_0* 
* 
* 
* 
Ц
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

Лtrace_0* 

Мtrace_0* 

j0
k1*

j0
k1*
* 
Ш
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

Тtrace_0* 

Уtrace_0* 
_Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

s0
t1*

s0
t1*
* 
Ш
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

Щtrace_0* 

Ъtrace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

аtrace_0* 

бtrace_0* 
* 
* 
* 
Щ
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses* 

зtrace_0* 

иtrace_0* 
* 
* 
* 
* 
Ь
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses* 

оtrace_0* 

пtrace_0* 
* 
* 
* 
Ь
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses* 

╡trace_0* 

╢trace_0* 

Х0
Ц1*

Х0
Ц1*
* 
Ю
╖non_trainable_variables
╕layers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses*

╝trace_0* 

╜trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses* 

├trace_0
─trace_1* 

┼trace_0
╞trace_1* 
* 

д0
е1*

д0
е1*
* 
Ю
╟non_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses*

╠trace_0* 

═trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 

╙trace_0
╘trace_1* 

╒trace_0
╓trace_1* 
* 

│0
┤1*

│0
┤1*
* 
Ю
╫non_trainable_variables
╪layers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
н	variables
оtrainable_variables
пregularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses*

▄trace_0* 

▌trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

╗0
╝1*

╗0
╝1*
* 
Ю
▐non_trainable_variables
▀layers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses*

уtrace_0* 

фtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
х0
ц1*
* 
* 
* 
* 
* 
* 
ы
╟0
ч1
ш2
щ3
ъ4
ы5
ь6
э7
ю8
я9
Ё10
ё11
Є12
є13
Ї14
ї15
Ў16
ў17
°18
∙19
·20
√21
№22
¤23
■24
 25
А26
Б27
В28
Г29
Д30
Е31
Ж32
З33
И34
Й35
К36
Л37
М38
Н39
О40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
о
ч0
щ1
ы2
э3
я4
ё5
є6
ї7
ў8
∙9
√10
¤11
 12
Б13
Г14
Е15
З16
Й17
Л18
Н19*
о
ш0
ъ1
ь2
ю3
Ё4
Є5
Ї6
Ў7
°8
·9
№10
■11
А12
В13
Д14
Ж15
И16
К17
М18
О19*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
П	variables
Р	keras_api

Сtotal

Тcount*
M
У	variables
Ф	keras_api

Хtotal

Цcount
Ч
_fn_kwargs*
_Y
VARIABLE_VALUEAdam/m/conv3d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv3d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv3d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv2d/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv2d/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_2/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_2/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_2/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_2/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*

С0
Т1*

П	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Х0
Ц1*

У	variables*
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
▓
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv2d/kernelconv2d/biasconv3d_1/kernelconv3d_1/biasconv2d_1/kernelconv2d_1/biasconv3d_2/kernelconv3d_2/biasconv2d_2/kernelconv2d_2/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense/kernel
dense/bias	iterationlearning_rateAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/biasAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/biasAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/biasAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/biasAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/biasAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biastotal_1count_1totalcountConst*O
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
__inference__traced_save_12808
н
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv2d/kernelconv2d/biasconv3d_1/kernelconv3d_1/biasconv2d_1/kernelconv2d_1/biasconv3d_2/kernelconv3d_2/biasconv2d_2/kernelconv2d_2/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense/kernel
dense/bias	iterationlearning_rateAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/biasAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/biasAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/biasAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/biasAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/biasAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biastotal_1count_1totalcount*N
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
!__inference__traced_restore_13015фл
│
№
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12151

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
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
:            X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:            i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:            S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$: : 20
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
:         $$
 
_user_specified_nameinputs
█
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_12349

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
┘
`
B__inference_dropout_layer_call_and_return_conditional_losses_11750

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_1_layer_call_fn_12140

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11547w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:            <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12136:%!

_user_specified_name12134:W S
/
_output_shapes
:         $$
 
_user_specified_nameinputs
█
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_11761

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
▓
I
-__inference_max_pooling2d_layer_call_fn_12106

inputs
identity╓
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
GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11431Г
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
О
Ы
&__inference_conv2d_layer_call_fn_12080

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_11513w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ll<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12076:%!

_user_specified_name12074:W S
/
_output_shapes
:         pp
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_1_layer_call_and_return_conditional_losses_11615

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
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
Т
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_11471

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
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12221

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
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
╒

Ў
B__inference_dense_2_layer_call_and_return_conditional_losses_11644

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
Я

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_12344

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
╘
■
A__inference_conv3d_layer_call_and_return_conditional_losses_11490

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         .>*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         .>\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:         .>m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         .>S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
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
п^
╣	
@__inference_model_layer_call_and_return_conditional_losses_11696
input_2
input_1*
conv3d_11491:
conv3d_11493:&
conv2d_11514:
conv2d_11516:,
conv3d_1_11530: 
conv3d_1_11532: (
conv2d_1_11548: 
conv2d_1_11550: ,
conv3d_2_11564:  
conv3d_2_11566: (
conv2d_2_11582: @
conv2d_2_11584:@!
dense_1_11616:
АА
dense_1_11618:	А!
dense_2_11645:
АА
dense_2_11647:	А!
dense_3_11674:
АА
dense_3_11676:	А
dense_11690:	А
dense_11692:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallЄ
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_11491conv3d_11493*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         .>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_11490└
identity/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         pp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_identity_layer_call_and_return_conditional_losses_11500ю
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         >* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11421И
conv2d/StatefulPartitionedCallStatefulPartitionedCall!identity/PartitionedCall:output:0conv2d_11514conv2d_11516*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_11513Щ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_11530conv3d_1_11532*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         < *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11529ъ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11431Ї
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11441Х
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_11548conv2d_1_11550*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11547Ы
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_11564conv3d_2_11566*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11563Ё
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11451Ї
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11461Ч
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_11582conv2d_2_11584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11581Ё
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_11471q
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          └  е
tf.reshape/ReshapeReshape(max_pooling3d_2/PartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         └Ж
concatenate/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0tf.reshape/Reshape:output:0*
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
F__inference_concatenate_layer_call_and_return_conditional_losses_11596╘
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11603Д
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_11616dense_1_11618*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_11615ш
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11632М
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_11645dense_2_11647*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_11644О
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_11661О
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_11674dense_3_11676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_11673Г
dense/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_11690dense_11692*
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11689u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╝
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:%!

_user_specified_name11692:%!

_user_specified_name11690:%!

_user_specified_name11676:%!

_user_specified_name11674:%!

_user_specified_name11647:%!

_user_specified_name11645:%!

_user_specified_name11618:%!

_user_specified_name11616:%!

_user_specified_name11584:%!

_user_specified_name11582:%!

_user_specified_name11566:%
!

_user_specified_name11564:%	!

_user_specified_name11550:%!

_user_specified_name11548:%!

_user_specified_name11532:%!

_user_specified_name11530:%!

_user_specified_name11516:%!

_user_specified_name11514:%!

_user_specified_name11493:%!

_user_specified_name11491:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_1:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_2
Х
╒
%__inference_model_layer_call_fn_11866
input_2
input_1%
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: '
	unknown_7:  
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:
АА

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А

unknown_18:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCallinput_2input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
@__inference_model_layer_call_and_return_conditional_losses_11774o
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

_user_specified_name11862:%!

_user_specified_name11860:%!

_user_specified_name11858:%!

_user_specified_name11856:%!

_user_specified_name11854:%!

_user_specified_name11852:%!

_user_specified_name11850:%!

_user_specified_name11848:%!

_user_specified_name11846:%!

_user_specified_name11844:%!

_user_specified_name11842:%
!

_user_specified_name11840:%	!

_user_specified_name11838:%!

_user_specified_name11836:%!

_user_specified_name11834:%!

_user_specified_name11832:%!

_user_specified_name11830:%!

_user_specified_name11828:%!

_user_specified_name11826:%!

_user_specified_name11824:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_1:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_2
╘
■
A__inference_conv3d_layer_call_and_return_conditional_losses_12071

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         .>*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         .>\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:         .>m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         .>S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
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
в
Я
&__inference_conv3d_layer_call_fn_12060

inputs%
unknown:
	unknown_0:
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         .>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_11490{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         .><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
0@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12056:%!

_user_specified_name12054:[ W
3
_output_shapes!
:         
0@
 
_user_specified_nameinputs
╓
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_12101

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
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
н
C
'__inference_flatten_layer_call_fn_12249

inputs
identityо
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
GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11603a
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
ъ
K
/__inference_max_pooling3d_2_layer_call_fn_12216

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
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11461Р
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
│
№
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11547

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
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
:            X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:            i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:            S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$: : 20
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
:         $$
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_3_layer_call_and_return_conditional_losses_12369

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
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
К}
ў
 __inference__wrapped_model_11416
input_2
input_1I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:E
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource: <
.model_conv3d_1_biasadd_readvariableop_resource: G
-model_conv2d_1_conv2d_readvariableop_resource: <
.model_conv2d_1_biasadd_readvariableop_resource: K
-model_conv3d_2_conv3d_readvariableop_resource:  <
.model_conv3d_2_biasadd_readvariableop_resource: G
-model_conv2d_2_conv2d_readvariableop_resource: @<
.model_conv2d_2_biasadd_readvariableop_resource:@@
,model_dense_1_matmul_readvariableop_resource:
АА<
-model_dense_1_biasadd_readvariableop_resource:	А@
,model_dense_2_matmul_readvariableop_resource:
АА<
-model_dense_2_biasadd_readvariableop_resource:	А@
,model_dense_3_matmul_readvariableop_resource:
АА<
-model_dense_3_biasadd_readvariableop_resource:	А=
*model_dense_matmul_readvariableop_resource:	А9
+model_dense_biasadd_readvariableop_resource:
identityИв#model/conv2d/BiasAdd/ReadVariableOpв"model/conv2d/Conv2D/ReadVariableOpв%model/conv2d_1/BiasAdd/ReadVariableOpв$model/conv2d_1/Conv2D/ReadVariableOpв%model/conv2d_2/BiasAdd/ReadVariableOpв$model/conv2d_2/Conv2D/ReadVariableOpв#model/conv3d/BiasAdd/ReadVariableOpв"model/conv3d/Conv3D/ReadVariableOpв%model/conv3d_1/BiasAdd/ReadVariableOpв$model/conv3d_1/Conv3D/ReadVariableOpв%model/conv3d_2/BiasAdd/ReadVariableOpв$model/conv3d_2/Conv3D/ReadVariableOpв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв$model/dense_2/BiasAdd/ReadVariableOpв#model/dense_2/MatMul/ReadVariableOpв$model/dense_3/BiasAdd/ReadVariableOpв#model/dense_3/MatMul/ReadVariableOpЪ
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0║
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         .>*
paddingVALID*
strides	
М
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0и
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         .>v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:         .>f
model/identity/IdentityIdentityinput_2*
T0*/
_output_shapes
:         pp╟
model/max_pooling3d/MaxPool3D	MaxPool3Dmodel/conv3d/Relu:activations:0*
T0*3
_output_shapes!
:         >*
ksize	
*
paddingVALID*
strides	
Ц
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╬
model/conv2d/Conv2DConv2D model/identity/Identity:output:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll*
paddingVALID*
strides
М
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         llr
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         llЮ
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0▌
model/conv3d_1/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         < *
paddingVALID*
strides	
Р
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0о
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         < z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:         < ┤
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*/
_output_shapes
:         $$*
ksize
*
paddingVALID*
strides
╦
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:         
 *
ksize	
*
paddingVALID*
strides	
Ъ
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╓
model/conv2d_1/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingVALID*
strides
Р
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            v
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:            Ю
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0▀
model/conv3d_2/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:          *
paddingVALID*
strides	
Р
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0о
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:          z
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:          ╕
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:         

 *
ksize
*
paddingVALID*
strides
╦
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:          *
ksize	
*
paddingVALID*
strides	
Ъ
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╪
model/conv2d_2/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
Р
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0к
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @v
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @╕
model/max_pooling2d_2/MaxPoolMaxPool!model/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
w
model/tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          └  ▒
model/tf.reshape/ReshapeReshape(model/max_pooling3d_2/MaxPool3D:output:0'model/tf.reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         └_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :█
model/concatenate/concatConcatV2&model/max_pooling2d_2/MaxPool:output:0!model/tf.reshape/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:         Аd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ф
model/flatten/ReshapeReshape!model/concatenate/concat:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:         АТ
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ю
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Аw
model/dropout/IdentityIdentity model/dense_1/Relu:activations:0*
T0*(
_output_shapes
:         АТ
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Я
model/dense_2/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         Аy
model/dropout_1/IdentityIdentity model/dense_2/Relu:activations:0*
T0*(
_output_shapes
:         АТ
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0б
model/dense_3/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         АН
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ы
model/dense/MatMulMatMul model/dense_3/Relu:activations:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         К
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         n
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         l
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         д
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp:($
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
_user_specified_name	input_1:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_2
╓
А
C__inference_conv3d_1_layer_call_and_return_conditional_losses_12131

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         < *
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         < \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:         < m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         < S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         >: : 20
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
:         >
 
_user_specified_nameinputs
╧
b
)__inference_dropout_1_layer_call_fn_12327

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
D__inference_dropout_1_layer_call_and_return_conditional_losses_11661p
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
╪
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11441

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
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
╟м
и)
!__inference__traced_restore_13015
file_prefix<
assignvariableop_conv3d_kernel:,
assignvariableop_1_conv3d_bias::
 assignvariableop_2_conv2d_kernel:,
assignvariableop_3_conv2d_bias:@
"assignvariableop_4_conv3d_1_kernel: .
 assignvariableop_5_conv3d_1_bias: <
"assignvariableop_6_conv2d_1_kernel: .
 assignvariableop_7_conv2d_1_bias: @
"assignvariableop_8_conv3d_2_kernel:  .
 assignvariableop_9_conv3d_2_bias: =
#assignvariableop_10_conv2d_2_kernel: @/
!assignvariableop_11_conv2d_2_bias:@6
"assignvariableop_12_dense_1_kernel:
АА/
 assignvariableop_13_dense_1_bias:	А6
"assignvariableop_14_dense_2_kernel:
АА/
 assignvariableop_15_dense_2_bias:	А6
"assignvariableop_16_dense_3_kernel:
АА/
 assignvariableop_17_dense_3_bias:	А3
 assignvariableop_18_dense_kernel:	А,
assignvariableop_19_dense_bias:'
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: F
(assignvariableop_22_adam_m_conv3d_kernel:F
(assignvariableop_23_adam_v_conv3d_kernel:4
&assignvariableop_24_adam_m_conv3d_bias:4
&assignvariableop_25_adam_v_conv3d_bias:B
(assignvariableop_26_adam_m_conv2d_kernel:B
(assignvariableop_27_adam_v_conv2d_kernel:4
&assignvariableop_28_adam_m_conv2d_bias:4
&assignvariableop_29_adam_v_conv2d_bias:H
*assignvariableop_30_adam_m_conv3d_1_kernel: H
*assignvariableop_31_adam_v_conv3d_1_kernel: 6
(assignvariableop_32_adam_m_conv3d_1_bias: 6
(assignvariableop_33_adam_v_conv3d_1_bias: D
*assignvariableop_34_adam_m_conv2d_1_kernel: D
*assignvariableop_35_adam_v_conv2d_1_kernel: 6
(assignvariableop_36_adam_m_conv2d_1_bias: 6
(assignvariableop_37_adam_v_conv2d_1_bias: H
*assignvariableop_38_adam_m_conv3d_2_kernel:  H
*assignvariableop_39_adam_v_conv3d_2_kernel:  6
(assignvariableop_40_adam_m_conv3d_2_bias: 6
(assignvariableop_41_adam_v_conv3d_2_bias: D
*assignvariableop_42_adam_m_conv2d_2_kernel: @D
*assignvariableop_43_adam_v_conv2d_2_kernel: @6
(assignvariableop_44_adam_m_conv2d_2_bias:@6
(assignvariableop_45_adam_v_conv2d_2_bias:@=
)assignvariableop_46_adam_m_dense_1_kernel:
АА=
)assignvariableop_47_adam_v_dense_1_kernel:
АА6
'assignvariableop_48_adam_m_dense_1_bias:	А6
'assignvariableop_49_adam_v_dense_1_bias:	А=
)assignvariableop_50_adam_m_dense_2_kernel:
АА=
)assignvariableop_51_adam_v_dense_2_kernel:
АА6
'assignvariableop_52_adam_m_dense_2_bias:	А6
'assignvariableop_53_adam_v_dense_2_bias:	А=
)assignvariableop_54_adam_m_dense_3_kernel:
АА=
)assignvariableop_55_adam_v_dense_3_kernel:
АА6
'assignvariableop_56_adam_m_dense_3_bias:	А6
'assignvariableop_57_adam_v_dense_3_bias:	А:
'assignvariableop_58_adam_m_dense_kernel:	А:
'assignvariableop_59_adam_v_dense_kernel:	А3
%assignvariableop_60_adam_m_dense_bias:3
%assignvariableop_61_adam_v_dense_bias:%
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
:╖
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*&
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
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*&
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
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_3_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*&
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
:┴
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_m_conv2d_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_v_conv2d_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_m_conv2d_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_v_conv2d_biasIdentity_29:output:0"/device:CPU:0*&
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
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_conv2d_1_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_conv2d_1_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_conv2d_1_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_conv2d_1_biasIdentity_37:output:0"/device:CPU:0*&
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
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_conv2d_2_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_conv2d_2_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_conv2d_2_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_conv2d_2_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_m_dense_1_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_v_dense_1_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_m_dense_1_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_v_dense_1_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_m_dense_2_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_v_dense_2_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_m_dense_2_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_v_dense_2_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_m_dense_3_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_v_dense_3_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_m_dense_3_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_v_dense_3_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_m_dense_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_v_dense_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_m_dense_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_61AssignVariableOp%assignvariableop_61_adam_v_dense_biasIdentity_61:output:0"/device:CPU:0*&
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
_user_specified_name	total_1:1>-
+
_user_specified_nameAdam/v/dense/bias:1=-
+
_user_specified_nameAdam/m/dense/bias:3</
-
_user_specified_nameAdam/v/dense/kernel:3;/
-
_user_specified_nameAdam/m/dense/kernel:3:/
-
_user_specified_nameAdam/v/dense_3/bias:39/
-
_user_specified_nameAdam/m/dense_3/bias:581
/
_user_specified_nameAdam/v/dense_3/kernel:571
/
_user_specified_nameAdam/m/dense_3/kernel:36/
-
_user_specified_nameAdam/v/dense_2/bias:35/
-
_user_specified_nameAdam/m/dense_2/bias:541
/
_user_specified_nameAdam/v/dense_2/kernel:531
/
_user_specified_nameAdam/m/dense_2/kernel:32/
-
_user_specified_nameAdam/v/dense_1/bias:31/
-
_user_specified_nameAdam/m/dense_1/bias:501
/
_user_specified_nameAdam/v/dense_1/kernel:5/1
/
_user_specified_nameAdam/m/dense_1/kernel:4.0
.
_user_specified_nameAdam/v/conv2d_2/bias:4-0
.
_user_specified_nameAdam/m/conv2d_2/bias:6,2
0
_user_specified_nameAdam/v/conv2d_2/kernel:6+2
0
_user_specified_nameAdam/m/conv2d_2/kernel:4*0
.
_user_specified_nameAdam/v/conv3d_2/bias:4)0
.
_user_specified_nameAdam/m/conv3d_2/bias:6(2
0
_user_specified_nameAdam/v/conv3d_2/kernel:6'2
0
_user_specified_nameAdam/m/conv3d_2/kernel:4&0
.
_user_specified_nameAdam/v/conv2d_1/bias:4%0
.
_user_specified_nameAdam/m/conv2d_1/bias:6$2
0
_user_specified_nameAdam/v/conv2d_1/kernel:6#2
0
_user_specified_nameAdam/m/conv2d_1/kernel:4"0
.
_user_specified_nameAdam/v/conv3d_1/bias:4!0
.
_user_specified_nameAdam/m/conv3d_1/bias:6 2
0
_user_specified_nameAdam/v/conv3d_1/kernel:62
0
_user_specified_nameAdam/m/conv3d_1/kernel:2.
,
_user_specified_nameAdam/v/conv2d/bias:2.
,
_user_specified_nameAdam/m/conv2d/bias:40
.
_user_specified_nameAdam/v/conv2d/kernel:40
.
_user_specified_nameAdam/m/conv2d/kernel:2.
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
_user_specified_name	iteration:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:-)
'
_user_specified_nameconv2d_2/bias:/+
)
_user_specified_nameconv2d_2/kernel:-
)
'
_user_specified_nameconv3d_2/bias:/	+
)
_user_specified_nameconv3d_2/kernel:-)
'
_user_specified_nameconv2d_1/bias:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:+'
%
_user_specified_nameconv2d/bias:-)
'
_user_specified_nameconv2d/kernel:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╦
`
'__inference_dropout_layer_call_fn_12280

inputs
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11632p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12171

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
B__inference_dense_2_layer_call_and_return_conditional_losses_12322

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
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
╪
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_12161

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
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
╓
А
C__inference_conv3d_2_layer_call_and_return_conditional_losses_12191

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:          *
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:          \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:          m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:          S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
 : : 20
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
:         
 
 
_user_specified_nameinputs
я
Ч
'__inference_dense_1_layer_call_fn_12264

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_11615p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12260:%!

_user_specified_name12258:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ж
б
(__inference_conv3d_1_layer_call_fn_12120

inputs%
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         < *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11529{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         < <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         >: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12116:%!

_user_specified_name12114:[ W
3
_output_shapes!
:         >
 
_user_specified_nameinputs
Э

a
B__inference_dropout_layer_call_and_return_conditional_losses_12297

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
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╪
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11461

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
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
╢
K
/__inference_max_pooling2d_2_layer_call_fn_12226

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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_11471Г
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
Р
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12111

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
▒
·
A__inference_conv2d_layer_call_and_return_conditional_losses_11513

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll*
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
:         llX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         lli
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         llS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
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
я
Ч
'__inference_dense_3_layer_call_fn_12358

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_11673p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12354:%!

_user_specified_name12352:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨

Є
@__inference_dense_layer_call_and_return_conditional_losses_12389

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
:         А: : 20
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
:         А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11451

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
▄Ё
╦<
__inference__traced_save_12808
file_prefixB
$read_disablecopyonread_conv3d_kernel:2
$read_1_disablecopyonread_conv3d_bias:@
&read_2_disablecopyonread_conv2d_kernel:2
$read_3_disablecopyonread_conv2d_bias:F
(read_4_disablecopyonread_conv3d_1_kernel: 4
&read_5_disablecopyonread_conv3d_1_bias: B
(read_6_disablecopyonread_conv2d_1_kernel: 4
&read_7_disablecopyonread_conv2d_1_bias: F
(read_8_disablecopyonread_conv3d_2_kernel:  4
&read_9_disablecopyonread_conv3d_2_bias: C
)read_10_disablecopyonread_conv2d_2_kernel: @5
'read_11_disablecopyonread_conv2d_2_bias:@<
(read_12_disablecopyonread_dense_1_kernel:
АА5
&read_13_disablecopyonread_dense_1_bias:	А<
(read_14_disablecopyonread_dense_2_kernel:
АА5
&read_15_disablecopyonread_dense_2_bias:	А<
(read_16_disablecopyonread_dense_3_kernel:
АА5
&read_17_disablecopyonread_dense_3_bias:	А9
&read_18_disablecopyonread_dense_kernel:	А2
$read_19_disablecopyonread_dense_bias:-
#read_20_disablecopyonread_iteration:	 1
'read_21_disablecopyonread_learning_rate: L
.read_22_disablecopyonread_adam_m_conv3d_kernel:L
.read_23_disablecopyonread_adam_v_conv3d_kernel::
,read_24_disablecopyonread_adam_m_conv3d_bias::
,read_25_disablecopyonread_adam_v_conv3d_bias:H
.read_26_disablecopyonread_adam_m_conv2d_kernel:H
.read_27_disablecopyonread_adam_v_conv2d_kernel::
,read_28_disablecopyonread_adam_m_conv2d_bias::
,read_29_disablecopyonread_adam_v_conv2d_bias:N
0read_30_disablecopyonread_adam_m_conv3d_1_kernel: N
0read_31_disablecopyonread_adam_v_conv3d_1_kernel: <
.read_32_disablecopyonread_adam_m_conv3d_1_bias: <
.read_33_disablecopyonread_adam_v_conv3d_1_bias: J
0read_34_disablecopyonread_adam_m_conv2d_1_kernel: J
0read_35_disablecopyonread_adam_v_conv2d_1_kernel: <
.read_36_disablecopyonread_adam_m_conv2d_1_bias: <
.read_37_disablecopyonread_adam_v_conv2d_1_bias: N
0read_38_disablecopyonread_adam_m_conv3d_2_kernel:  N
0read_39_disablecopyonread_adam_v_conv3d_2_kernel:  <
.read_40_disablecopyonread_adam_m_conv3d_2_bias: <
.read_41_disablecopyonread_adam_v_conv3d_2_bias: J
0read_42_disablecopyonread_adam_m_conv2d_2_kernel: @J
0read_43_disablecopyonread_adam_v_conv2d_2_kernel: @<
.read_44_disablecopyonread_adam_m_conv2d_2_bias:@<
.read_45_disablecopyonread_adam_v_conv2d_2_bias:@C
/read_46_disablecopyonread_adam_m_dense_1_kernel:
ААC
/read_47_disablecopyonread_adam_v_dense_1_kernel:
АА<
-read_48_disablecopyonread_adam_m_dense_1_bias:	А<
-read_49_disablecopyonread_adam_v_dense_1_bias:	АC
/read_50_disablecopyonread_adam_m_dense_2_kernel:
ААC
/read_51_disablecopyonread_adam_v_dense_2_kernel:
АА<
-read_52_disablecopyonread_adam_m_dense_2_bias:	А<
-read_53_disablecopyonread_adam_v_dense_2_bias:	АC
/read_54_disablecopyonread_adam_m_dense_3_kernel:
ААC
/read_55_disablecopyonread_adam_v_dense_3_kernel:
АА<
-read_56_disablecopyonread_adam_m_dense_3_bias:	А<
-read_57_disablecopyonread_adam_v_dense_3_bias:	А@
-read_58_disablecopyonread_adam_m_dense_kernel:	А@
-read_59_disablecopyonread_adam_v_dense_kernel:	А9
+read_60_disablecopyonread_adam_m_dense_bias:9
+read_61_disablecopyonread_adam_v_dense_bias:+
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
:*
dtype0u
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:m

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0**
_output_shapes
:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv3d_bias"/device:CPU:0*
_output_shapes
 а
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv3d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 о
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_conv2d_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
 а
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_conv2d_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv3d_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0**
_output_shapes
: *
dtype0y

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
: o

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0**
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv3d_1_bias"/device:CPU:0*
_output_shapes
 в
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv3d_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 ░
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 в
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_conv3d_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:  *
dtype0z
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:  q
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0**
_output_shapes
:  z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_conv3d_2_bias"/device:CPU:0*
_output_shapes
 в
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_conv3d_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 │
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_conv2d_2_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
: @|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 е
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_conv2d_2_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 м
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 е
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 м
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_dense_2_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_15/DisableCopyOnReadDisableCopyOnRead&read_15_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 е
Read_15/ReadVariableOpReadVariableOp&read_15_disablecopyonread_dense_2_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 м
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_dense_3_kernel^Read_16/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 е
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_dense_3_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:А{
Read_18/DisableCopyOnReadDisableCopyOnRead&read_18_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 й
Read_18/ReadVariableOpReadVariableOp&read_18_disablecopyonread_dense_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	Аy
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 в
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_dense_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:*
dtype0{
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0**
_output_shapes
:Г
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_adam_v_conv3d_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_adam_v_conv3d_kernel^Read_23/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0**
_output_shapes
:Б
Read_24/DisableCopyOnReadDisableCopyOnRead,read_24_disablecopyonread_adam_m_conv3d_bias"/device:CPU:0*
_output_shapes
 к
Read_24/ReadVariableOpReadVariableOp,read_24_disablecopyonread_adam_m_conv3d_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_25/DisableCopyOnReadDisableCopyOnRead,read_25_disablecopyonread_adam_v_conv3d_bias"/device:CPU:0*
_output_shapes
 к
Read_25/ReadVariableOpReadVariableOp,read_25_disablecopyonread_adam_v_conv3d_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
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
:Г
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_m_conv2d_kernel"/device:CPU:0*
_output_shapes
 ╕
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_m_conv2d_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
:Г
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_v_conv2d_kernel"/device:CPU:0*
_output_shapes
 ╕
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_v_conv2d_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
:Б
Read_28/DisableCopyOnReadDisableCopyOnRead,read_28_disablecopyonread_adam_m_conv2d_bias"/device:CPU:0*
_output_shapes
 к
Read_28/ReadVariableOpReadVariableOp,read_28_disablecopyonread_adam_m_conv2d_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
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
:Б
Read_29/DisableCopyOnReadDisableCopyOnRead,read_29_disablecopyonread_adam_v_conv2d_bias"/device:CPU:0*
_output_shapes
 к
Read_29/ReadVariableOpReadVariableOp,read_29_disablecopyonread_adam_v_conv2d_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
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
:Е
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 ╛
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_conv3d_1_kernel^Read_30/DisableCopyOnRead"/device:CPU:0**
_output_shapes
: *
dtype0{
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
: q
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0**
_output_shapes
: Е
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 ╛
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_conv3d_1_kernel^Read_31/DisableCopyOnRead"/device:CPU:0**
_output_shapes
: *
dtype0{
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
: q
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0**
_output_shapes
: Г
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_conv3d_1_bias"/device:CPU:0*
_output_shapes
 м
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_conv3d_1_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_conv3d_1_bias"/device:CPU:0*
_output_shapes
 м
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_conv3d_1_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 ║
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_conv2d_1_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
: Е
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 ║
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_conv2d_1_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_conv2d_1_bias"/device:CPU:0*
_output_shapes
 м
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_conv2d_1_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_conv2d_1_bias"/device:CPU:0*
_output_shapes
 м
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_conv2d_1_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 ╛
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_conv3d_2_kernel^Read_38/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:  *
dtype0{
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:  q
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0**
_output_shapes
:  Е
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 ╛
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_conv3d_2_kernel^Read_39/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:  *
dtype0{
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:  q
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0**
_output_shapes
:  Г
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_conv3d_2_bias"/device:CPU:0*
_output_shapes
 м
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_conv3d_2_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_conv3d_2_bias"/device:CPU:0*
_output_shapes
 м
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_conv3d_2_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 ║
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_conv2d_2_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Е
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 ║
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_conv2d_2_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_m_conv2d_2_bias"/device:CPU:0*
_output_shapes
 м
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_m_conv2d_2_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_v_conv2d_2_bias"/device:CPU:0*
_output_shapes
 м
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_v_conv2d_2_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_46/DisableCopyOnReadDisableCopyOnRead/read_46_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 │
Read_46/ReadVariableOpReadVariableOp/read_46_disablecopyonread_adam_m_dense_1_kernel^Read_46/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_47/DisableCopyOnReadDisableCopyOnRead/read_47_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 │
Read_47/ReadVariableOpReadVariableOp/read_47_disablecopyonread_adam_v_dense_1_kernel^Read_47/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_48/DisableCopyOnReadDisableCopyOnRead-read_48_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 м
Read_48/ReadVariableOpReadVariableOp-read_48_disablecopyonread_adam_m_dense_1_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_49/DisableCopyOnReadDisableCopyOnRead-read_49_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 м
Read_49/ReadVariableOpReadVariableOp-read_49_disablecopyonread_adam_v_dense_1_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_50/DisableCopyOnReadDisableCopyOnRead/read_50_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 │
Read_50/ReadVariableOpReadVariableOp/read_50_disablecopyonread_adam_m_dense_2_kernel^Read_50/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_51/DisableCopyOnReadDisableCopyOnRead/read_51_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 │
Read_51/ReadVariableOpReadVariableOp/read_51_disablecopyonread_adam_v_dense_2_kernel^Read_51/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_52/DisableCopyOnReadDisableCopyOnRead-read_52_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 м
Read_52/ReadVariableOpReadVariableOp-read_52_disablecopyonread_adam_m_dense_2_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
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
Read_53/DisableCopyOnReadDisableCopyOnRead-read_53_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 м
Read_53/ReadVariableOpReadVariableOp-read_53_disablecopyonread_adam_v_dense_2_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
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
Read_54/DisableCopyOnReadDisableCopyOnRead/read_54_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 │
Read_54/ReadVariableOpReadVariableOp/read_54_disablecopyonread_adam_m_dense_3_kernel^Read_54/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_55/DisableCopyOnReadDisableCopyOnRead/read_55_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 │
Read_55/ReadVariableOpReadVariableOp/read_55_disablecopyonread_adam_v_dense_3_kernel^Read_55/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_56/DisableCopyOnReadDisableCopyOnRead-read_56_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 м
Read_56/ReadVariableOpReadVariableOp-read_56_disablecopyonread_adam_m_dense_3_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_57/DisableCopyOnReadDisableCopyOnRead-read_57_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 м
Read_57/ReadVariableOpReadVariableOp-read_57_disablecopyonread_adam_v_dense_3_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_58/DisableCopyOnReadDisableCopyOnRead-read_58_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 ░
Read_58/ReadVariableOpReadVariableOp-read_58_disablecopyonread_adam_m_dense_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0q
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аh
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_59/DisableCopyOnReadDisableCopyOnRead-read_59_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 ░
Read_59/ReadVariableOpReadVariableOp-read_59_disablecopyonread_adam_v_dense_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0q
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аh
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:	АА
Read_60/DisableCopyOnReadDisableCopyOnRead+read_60_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 й
Read_60/ReadVariableOpReadVariableOp+read_60_disablecopyonread_adam_m_dense_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
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
:А
Read_61/DisableCopyOnReadDisableCopyOnRead+read_61_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 й
Read_61/ReadVariableOpReadVariableOp+read_61_disablecopyonread_adam_v_dense_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
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
_user_specified_name	total_1:1>-
+
_user_specified_nameAdam/v/dense/bias:1=-
+
_user_specified_nameAdam/m/dense/bias:3</
-
_user_specified_nameAdam/v/dense/kernel:3;/
-
_user_specified_nameAdam/m/dense/kernel:3:/
-
_user_specified_nameAdam/v/dense_3/bias:39/
-
_user_specified_nameAdam/m/dense_3/bias:581
/
_user_specified_nameAdam/v/dense_3/kernel:571
/
_user_specified_nameAdam/m/dense_3/kernel:36/
-
_user_specified_nameAdam/v/dense_2/bias:35/
-
_user_specified_nameAdam/m/dense_2/bias:541
/
_user_specified_nameAdam/v/dense_2/kernel:531
/
_user_specified_nameAdam/m/dense_2/kernel:32/
-
_user_specified_nameAdam/v/dense_1/bias:31/
-
_user_specified_nameAdam/m/dense_1/bias:501
/
_user_specified_nameAdam/v/dense_1/kernel:5/1
/
_user_specified_nameAdam/m/dense_1/kernel:4.0
.
_user_specified_nameAdam/v/conv2d_2/bias:4-0
.
_user_specified_nameAdam/m/conv2d_2/bias:6,2
0
_user_specified_nameAdam/v/conv2d_2/kernel:6+2
0
_user_specified_nameAdam/m/conv2d_2/kernel:4*0
.
_user_specified_nameAdam/v/conv3d_2/bias:4)0
.
_user_specified_nameAdam/m/conv3d_2/bias:6(2
0
_user_specified_nameAdam/v/conv3d_2/kernel:6'2
0
_user_specified_nameAdam/m/conv3d_2/kernel:4&0
.
_user_specified_nameAdam/v/conv2d_1/bias:4%0
.
_user_specified_nameAdam/m/conv2d_1/bias:6$2
0
_user_specified_nameAdam/v/conv2d_1/kernel:6#2
0
_user_specified_nameAdam/m/conv2d_1/kernel:4"0
.
_user_specified_nameAdam/v/conv3d_1/bias:4!0
.
_user_specified_nameAdam/m/conv3d_1/bias:6 2
0
_user_specified_nameAdam/v/conv3d_1/kernel:62
0
_user_specified_nameAdam/m/conv3d_1/kernel:2.
,
_user_specified_nameAdam/v/conv2d/bias:2.
,
_user_specified_nameAdam/m/conv2d/bias:40
.
_user_specified_nameAdam/v/conv2d/kernel:40
.
_user_specified_nameAdam/m/conv2d/kernel:2.
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
_user_specified_name	iteration:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:-)
'
_user_specified_nameconv2d_2/bias:/+
)
_user_specified_nameconv2d_2/kernel:-
)
'
_user_specified_nameconv3d_2/bias:/	+
)
_user_specified_nameconv3d_2/kernel:-)
'
_user_specified_nameconv2d_1/bias:/+
)
_user_specified_nameconv2d_1/kernel:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:+'
%
_user_specified_nameconv2d/bias:-)
'
_user_specified_nameconv2d/kernel:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ў
a
C__inference_identity_layer_call_and_return_conditional_losses_11500

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         ppc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         pp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         pp:W S
/
_output_shapes
:         pp
 
_user_specified_nameinputs
╞
^
B__inference_flatten_layer_call_and_return_conditional_losses_12255

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
│
№
C__inference_conv2d_2_layer_call_and_return_conditional_losses_12211

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

 : : 20
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
:         

 
 
_user_specified_nameinputs
ж
б
(__inference_conv3d_2_layer_call_fn_12180

inputs%
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11563{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
 : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12176:%!

_user_specified_name12174:[ W
3
_output_shapes!
:         
 
 
_user_specified_nameinputs
ы
p
F__inference_concatenate_layer_call_and_return_conditional_losses_11596

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
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @:         └:XT
0
_output_shapes
:         └
 
_user_specified_nameinputs:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12231

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
є
╙
#__inference_signature_wrapper_12041
input_1
input_2%
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: '
	unknown_7:  
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:
АА

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А

unknown_18:
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinput_2input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 __inference__wrapped_model_11416o
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

_user_specified_name12037:%!

_user_specified_name12035:%!

_user_specified_name12033:%!

_user_specified_name12031:%!

_user_specified_name12029:%!

_user_specified_name12027:%!

_user_specified_name12025:%!

_user_specified_name12023:%!

_user_specified_name12021:%!

_user_specified_name12019:%!

_user_specified_name12017:%
!

_user_specified_name12015:%	!

_user_specified_name12013:%!

_user_specified_name12011:%!

_user_specified_name12009:%!

_user_specified_name12007:%!

_user_specified_name12005:%!

_user_specified_name12003:%!

_user_specified_name12001:%!

_user_specified_name11999:XT
/
_output_shapes
:         pp
!
_user_specified_name	input_2:\ X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_1
┌
W
+__inference_concatenate_layer_call_fn_12237
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
F__inference_concatenate_layer_call_and_return_conditional_losses_11596i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @:         └:ZV
0
_output_shapes
:         └
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs_0
╒

Ў
B__inference_dense_1_layer_call_and_return_conditional_losses_12275

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
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
б
E
)__inference_dropout_1_layer_call_fn_12332

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
D__inference_dropout_1_layer_call_and_return_conditional_losses_11761a
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
'__inference_dense_2_layer_call_fn_12311

inputs
unknown:
АА
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
B__inference_dense_2_layer_call_and_return_conditional_losses_11644p
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
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12307:%!

_user_specified_name12305:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨

Є
@__inference_dense_layer_call_and_return_conditional_losses_11689

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
:         А: : 20
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
:         А
 
_user_specified_nameinputs
Я

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_11661

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
│
№
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11581

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

 : : 20
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
:         

 
 
_user_specified_nameinputs
╓
А
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11563

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:          *
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:          \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:          m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:          S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         
 : : 20
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
:         
 
 
_user_specified_nameinputs
Э

a
B__inference_dropout_layer_call_and_return_conditional_losses_11632

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
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╞
^
B__inference_flatten_layer_call_and_return_conditional_losses_11603

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
Э
C
'__inference_dropout_layer_call_fn_12285

inputs
identityо
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
GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11750a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_2_layer_call_fn_12200

inputs!
unknown: @
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
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11581w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

 : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12196:%!

_user_specified_name12194:W S
/
_output_shapes
:         

 
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_1_layer_call_fn_12166

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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11451Г
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
╓
А
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11529

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         < *
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         < \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:         < m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         < S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         >: : 20
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
:         >
 
_user_specified_nameinputs
ц
I
-__inference_max_pooling3d_layer_call_fn_12096

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
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11421Р
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
Ї
r
F__inference_concatenate_layer_call_and_return_conditional_losses_12244
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
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @:         └:ZV
0
_output_shapes
:         └
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs_0
╒

Ў
B__inference_dense_3_layer_call_and_return_conditional_losses_11673

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
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
┘
`
B__inference_dropout_layer_call_and_return_conditional_losses_12302

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┼[
є
@__inference_model_layer_call_and_return_conditional_losses_11774
input_2
input_1*
conv3d_11700:
conv3d_11702:&
conv2d_11707:
conv2d_11709:,
conv3d_1_11712: 
conv3d_1_11714: (
conv2d_1_11719: 
conv2d_1_11721: ,
conv3d_2_11724:  
conv3d_2_11726: (
conv2d_2_11731: @
conv2d_2_11733:@!
dense_1_11741:
АА
dense_1_11743:	А!
dense_2_11752:
АА
dense_2_11754:	А!
dense_3_11763:
АА
dense_3_11765:	А
dense_11768:	А
dense_11770:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallЄ
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_11700conv3d_11702*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         .>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_11490└
identity/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         pp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_identity_layer_call_and_return_conditional_losses_11500ю
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         >* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11421И
conv2d/StatefulPartitionedCallStatefulPartitionedCall!identity/PartitionedCall:output:0conv2d_11707conv2d_11709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_11513Щ
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_1_11712conv3d_1_11714*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         < *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11529ъ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11431Ї
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11441Х
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_11719conv2d_1_11721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_11547Ы
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_2_11724conv3d_2_11726*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11563Ё
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11451Ї
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11461Ч
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_11731conv2d_2_11733*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_11581Ё
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_11471q
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          └  е
tf.reshape/ReshapeReshape(max_pooling3d_2/PartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         └Ж
concatenate/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0tf.reshape/Reshape:output:0*
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
F__inference_concatenate_layer_call_and_return_conditional_losses_11596╘
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11603Д
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_11741dense_1_11743*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_11615╪
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11750Д
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_11752dense_2_11754*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_11644▄
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_11761Ж
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_11763dense_3_11765*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_11673Г
dense/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_11768dense_11770*
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11689u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ў
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:         pp:         
0@: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:%!

_user_specified_name11770:%!

_user_specified_name11768:%!

_user_specified_name11765:%!

_user_specified_name11763:%!

_user_specified_name11754:%!

_user_specified_name11752:%!

_user_specified_name11743:%!

_user_specified_name11741:%!

_user_specified_name11733:%!

_user_specified_name11731:%!

_user_specified_name11726:%
!

_user_specified_name11724:%	!

_user_specified_name11721:%!

_user_specified_name11719:%!

_user_specified_name11714:%!

_user_specified_name11712:%!

_user_specified_name11709:%!

_user_specified_name11707:%!

_user_specified_name11702:%!

_user_specified_name11700:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_1:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_2
╓
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11421

inputs
identity╛
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
К
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
Ў
a
C__inference_identity_layer_call_and_return_conditional_losses_12051

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         ppc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         pp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         pp:W S
/
_output_shapes
:         pp
 
_user_specified_nameinputs
ъ
K
/__inference_max_pooling3d_1_layer_call_fn_12156

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
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11441Р
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
Р
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_11431

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
Х
╒
%__inference_model_layer_call_fn_11820
input_2
input_1%
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6: '
	unknown_7:  
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:
АА

unknown_12:	А

unknown_13:
АА

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А

unknown_18:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCallinput_2input_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
@__inference_model_layer_call_and_return_conditional_losses_11696o
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

_user_specified_name11816:%!

_user_specified_name11814:%!

_user_specified_name11812:%!

_user_specified_name11810:%!

_user_specified_name11808:%!

_user_specified_name11806:%!

_user_specified_name11804:%!

_user_specified_name11802:%!

_user_specified_name11800:%!

_user_specified_name11798:%!

_user_specified_name11796:%
!

_user_specified_name11794:%	!

_user_specified_name11792:%!

_user_specified_name11790:%!

_user_specified_name11788:%!

_user_specified_name11786:%!

_user_specified_name11784:%!

_user_specified_name11782:%!

_user_specified_name11780:%!

_user_specified_name11778:\X
3
_output_shapes!
:         
0@
!
_user_specified_name	input_1:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_2
▒
·
A__inference_conv2d_layer_call_and_return_conditional_losses_12091

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll*
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
:         llX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         lli
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         llS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
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
ч
У
%__inference_dense_layer_call_fn_12378

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╒
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
GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11689o
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
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name12374:%!

_user_specified_name12372:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╗
D
(__inference_identity_layer_call_fn_12046

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         pp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_identity_layer_call_and_return_conditional_losses_11500h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         pp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         pp:W S
/
_output_shapes
:         pp
 
_user_specified_nameinputs"зL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*∙
serving_defaultх
G
input_1<
serving_default_input_1:0         
0@
C
input_28
serving_default_input_2:0         pp9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:√┐
н
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-6
layer-18
layer-19
layer_with_weights-7
layer-20
layer-21
layer_with_weights-8
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
е
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias
 0_jit_compiled_convolution_op"
_tf_keras_layer
▌
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
е
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
е
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op"
_tf_keras_layer
▌
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op"
_tf_keras_layer
е
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
е
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias
 l_jit_compiled_convolution_op"
_tf_keras_layer
▌
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op"
_tf_keras_layer
е
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
з
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
)
В	keras_api"
_tf_keras_layer
л
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
├
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses
Хkernel
	Цbias"
_tf_keras_layer
├
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
Э_random_generator"
_tf_keras_layer
├
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses
дkernel
	еbias"
_tf_keras_layer
├
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses
м_random_generator"
_tf_keras_layer
├
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses
│kernel
	┤bias"
_tf_keras_layer
├
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗kernel
	╝bias"
_tf_keras_layer
╛
.0
/1
72
83
L4
M5
U6
V7
j8
k9
s10
t11
Х12
Ц13
д14
е15
│16
┤17
╗18
╝19"
trackable_list_wrapper
╛
.0
/1
72
83
L4
M5
U6
V7
j8
k9
s10
t11
Х12
Ц13
д14
е15
│16
┤17
╗18
╝19"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┴
┬trace_0
├trace_12Ж
%__inference_model_layer_call_fn_11820
%__inference_model_layer_call_fn_11866╡
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
 z┬trace_0z├trace_1
ў
─trace_0
┼trace_12╝
@__inference_model_layer_call_and_return_conditional_losses_11696
@__inference_model_layer_call_and_return_conditional_losses_11774╡
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
 z─trace_0z┼trace_1
╘B╤
 __inference__wrapped_model_11416input_2input_1"Ш
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
╞
_variables
╟_iterations
╚_learning_rate
╔_index_dict
╩
_momentums
╦_velocities
╠_update_step_xla"
experimentalOptimizer
-
═serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ф
╙trace_02┼
(__inference_identity_layer_call_fn_12046Ш
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
 z╙trace_0
 
╘trace_02р
C__inference_identity_layer_call_and_return_conditional_losses_12051Ш
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
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╒non_trainable_variables
╓layers
╫metrics
 ╪layer_regularization_losses
┘layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
т
┌trace_02├
&__inference_conv3d_layer_call_fn_12060Ш
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
 z┌trace_0
¤
█trace_02▐
A__inference_conv3d_layer_call_and_return_conditional_losses_12071Ш
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
+:)2conv3d/kernel
:2conv3d/bias
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
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
т
сtrace_02├
&__inference_conv2d_layer_call_fn_12080Ш
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
 zсtrace_0
¤
тtrace_02▐
A__inference_conv2d_layer_call_and_return_conditional_losses_12091Ш
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
':%2conv2d/kernel
:2conv2d/bias
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
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
щ
шtrace_02╩
-__inference_max_pooling3d_layer_call_fn_12096Ш
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
 zшtrace_0
Д
щtrace_02х
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_12101Ш
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
щ
яtrace_02╩
-__inference_max_pooling2d_layer_call_fn_12106Ш
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
 zяtrace_0
Д
Ёtrace_02х
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12111Ш
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
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
ф
Ўtrace_02┼
(__inference_conv3d_1_layer_call_fn_12120Ш
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
 zЎtrace_0
 
ўtrace_02р
C__inference_conv3d_1_layer_call_and_return_conditional_losses_12131Ш
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
-:+ 2conv3d_1/kernel
: 2conv3d_1/bias
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
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
ф
¤trace_02┼
(__inference_conv2d_1_layer_call_fn_12140Ш
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
 z¤trace_0
 
■trace_02р
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12151Ш
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
):' 2conv2d_1/kernel
: 2conv2d_1/bias
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
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
ы
Дtrace_02╠
/__inference_max_pooling3d_1_layer_call_fn_12156Ш
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
 zДtrace_0
Ж
Еtrace_02ч
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_12161Ш
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ы
Лtrace_02╠
/__inference_max_pooling2d_1_layer_call_fn_12166Ш
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
 zЛtrace_0
Ж
Мtrace_02ч
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12171Ш
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
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ф
Тtrace_02┼
(__inference_conv3d_2_layer_call_fn_12180Ш
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
 zТtrace_0
 
Уtrace_02р
C__inference_conv3d_2_layer_call_and_return_conditional_losses_12191Ш
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
-:+  2conv3d_2/kernel
: 2conv3d_2/bias
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
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
ф
Щtrace_02┼
(__inference_conv2d_2_layer_call_fn_12200Ш
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
 zЩtrace_0
 
Ъtrace_02р
C__inference_conv2d_2_layer_call_and_return_conditional_losses_12211Ш
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
):' @2conv2d_2/kernel
:@2conv2d_2/bias
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
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
ы
аtrace_02╠
/__inference_max_pooling3d_2_layer_call_fn_12216Ш
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
 zаtrace_0
Ж
бtrace_02ч
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12221Ш
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
ы
зtrace_02╠
/__inference_max_pooling2d_2_layer_call_fn_12226Ш
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
 zзtrace_0
Ж
иtrace_02ч
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12231Ш
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
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
ч
оtrace_02╚
+__inference_concatenate_layer_call_fn_12237Ш
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
 zоtrace_0
В
пtrace_02у
F__inference_concatenate_layer_call_and_return_conditional_losses_12244Ш
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
у
╡trace_02─
'__inference_flatten_layer_call_fn_12249Ш
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
 z╡trace_0
■
╢trace_02▀
B__inference_flatten_layer_call_and_return_conditional_losses_12255Ш
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
0
Х0
Ц1"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╖non_trainable_variables
╕layers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
у
╝trace_02─
'__inference_dense_1_layer_call_fn_12264Ш
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
 z╝trace_0
■
╜trace_02▀
B__inference_dense_1_layer_call_and_return_conditional_losses_12275Ш
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
 z╜trace_0
": 
АА2dense_1/kernel
:А2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
╣
├trace_0
─trace_12■
'__inference_dropout_layer_call_fn_12280
'__inference_dropout_layer_call_fn_12285й
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
 z├trace_0z─trace_1
я
┼trace_0
╞trace_12┤
B__inference_dropout_layer_call_and_return_conditional_losses_12297
B__inference_dropout_layer_call_and_return_conditional_losses_12302й
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
 z┼trace_0z╞trace_1
"
_generic_user_object
0
д0
е1"
trackable_list_wrapper
0
д0
е1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╟non_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
у
╠trace_02─
'__inference_dense_2_layer_call_fn_12311Ш
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
 z╠trace_0
■
═trace_02▀
B__inference_dense_2_layer_call_and_return_conditional_losses_12322Ш
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
 z═trace_0
": 
АА2dense_2/kernel
:А2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
╜
╙trace_0
╘trace_12В
)__inference_dropout_1_layer_call_fn_12327
)__inference_dropout_1_layer_call_fn_12332й
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
 z╙trace_0z╘trace_1
є
╒trace_0
╓trace_12╕
D__inference_dropout_1_layer_call_and_return_conditional_losses_12344
D__inference_dropout_1_layer_call_and_return_conditional_losses_12349й
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
 z╒trace_0z╓trace_1
"
_generic_user_object
0
│0
┤1"
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╫non_trainable_variables
╪layers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
н	variables
оtrainable_variables
пregularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
у
▄trace_02─
'__inference_dense_3_layer_call_fn_12358Ш
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
 z▄trace_0
■
▌trace_02▀
B__inference_dense_3_layer_call_and_return_conditional_losses_12369Ш
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
 z▌trace_0
": 
АА2dense_3/kernel
:А2dense_3/bias
0
╗0
╝1"
trackable_list_wrapper
0
╗0
╝1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▐non_trainable_variables
▀layers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
с
уtrace_02┬
%__inference_dense_layer_call_fn_12378Ш
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
 zуtrace_0
№
фtrace_02▌
@__inference_dense_layer_call_and_return_conditional_losses_12389Ш
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
 zфtrace_0
:	А2dense/kernel
:2
dense/bias
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
х0
ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
%__inference_model_layer_call_fn_11820input_2input_1"м
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
%__inference_model_layer_call_fn_11866input_2input_1"м
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
@__inference_model_layer_call_and_return_conditional_losses_11696input_2input_1"м
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
@__inference_model_layer_call_and_return_conditional_losses_11774input_2input_1"м
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
╟0
ч1
ш2
щ3
ъ4
ы5
ь6
э7
ю8
я9
Ё10
ё11
Є12
є13
Ї14
ї15
Ў16
ў17
°18
∙19
·20
√21
№22
¤23
■24
 25
А26
Б27
В28
Г29
Д30
Е31
Ж32
З33
И34
Й35
К36
Л37
М38
Н39
О40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
╩
ч0
щ1
ы2
э3
я4
ё5
є6
ї7
ў8
∙9
√10
¤11
 12
Б13
Г14
Е15
З16
Й17
Л18
Н19"
trackable_list_wrapper
╩
ш0
ъ1
ь2
ю3
Ё4
Є5
Ї6
Ў7
°8
·9
№10
■11
А12
В13
Д14
Ж15
И16
К17
М18
О19"
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
#__inference_signature_wrapper_12041input_1input_2"д
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
	jinput_1
	jinput_2
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
(__inference_identity_layer_call_fn_12046inputs"Ш
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
C__inference_identity_layer_call_and_return_conditional_losses_12051inputs"Ш
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
╨B═
&__inference_conv3d_layer_call_fn_12060inputs"Ш
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
A__inference_conv3d_layer_call_and_return_conditional_losses_12071inputs"Ш
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
╨B═
&__inference_conv2d_layer_call_fn_12080inputs"Ш
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
A__inference_conv2d_layer_call_and_return_conditional_losses_12091inputs"Ш
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
-__inference_max_pooling3d_layer_call_fn_12096inputs"Ш
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
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_12101inputs"Ш
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
-__inference_max_pooling2d_layer_call_fn_12106inputs"Ш
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12111inputs"Ш
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
(__inference_conv3d_1_layer_call_fn_12120inputs"Ш
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
C__inference_conv3d_1_layer_call_and_return_conditional_losses_12131inputs"Ш
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
(__inference_conv2d_1_layer_call_fn_12140inputs"Ш
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12151inputs"Ш
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
/__inference_max_pooling3d_1_layer_call_fn_12156inputs"Ш
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
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_12161inputs"Ш
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
/__inference_max_pooling2d_1_layer_call_fn_12166inputs"Ш
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12171inputs"Ш
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
(__inference_conv3d_2_layer_call_fn_12180inputs"Ш
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
C__inference_conv3d_2_layer_call_and_return_conditional_losses_12191inputs"Ш
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
(__inference_conv2d_2_layer_call_fn_12200inputs"Ш
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_12211inputs"Ш
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
/__inference_max_pooling3d_2_layer_call_fn_12216inputs"Ш
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
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12221inputs"Ш
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
/__inference_max_pooling2d_2_layer_call_fn_12226inputs"Ш
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12231inputs"Ш
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
+__inference_concatenate_layer_call_fn_12237inputs_0inputs_1"Ш
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
F__inference_concatenate_layer_call_and_return_conditional_losses_12244inputs_0inputs_1"Ш
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
'__inference_flatten_layer_call_fn_12249inputs"Ш
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
B__inference_flatten_layer_call_and_return_conditional_losses_12255inputs"Ш
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
'__inference_dense_1_layer_call_fn_12264inputs"Ш
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
B__inference_dense_1_layer_call_and_return_conditional_losses_12275inputs"Ш
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
▌B┌
'__inference_dropout_layer_call_fn_12280inputs"д
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
▌B┌
'__inference_dropout_layer_call_fn_12285inputs"д
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
°Bї
B__inference_dropout_layer_call_and_return_conditional_losses_12297inputs"д
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
°Bї
B__inference_dropout_layer_call_and_return_conditional_losses_12302inputs"д
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
'__inference_dense_2_layer_call_fn_12311inputs"Ш
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
B__inference_dense_2_layer_call_and_return_conditional_losses_12322inputs"Ш
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
)__inference_dropout_1_layer_call_fn_12327inputs"д
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
)__inference_dropout_1_layer_call_fn_12332inputs"д
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_12344inputs"д
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_12349inputs"д
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
'__inference_dense_3_layer_call_fn_12358inputs"Ш
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
B__inference_dense_3_layer_call_and_return_conditional_losses_12369inputs"Ш
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
╧B╠
%__inference_dense_layer_call_fn_12378inputs"Ш
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
ъBч
@__inference_dense_layer_call_and_return_conditional_losses_12389inputs"Ш
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
П	variables
Р	keras_api

Сtotal

Тcount"
_tf_keras_metric
c
У	variables
Ф	keras_api

Хtotal

Цcount
Ч
_fn_kwargs"
_tf_keras_metric
0:.2Adam/m/conv3d/kernel
0:.2Adam/v/conv3d/kernel
:2Adam/m/conv3d/bias
:2Adam/v/conv3d/bias
,:*2Adam/m/conv2d/kernel
,:*2Adam/v/conv2d/kernel
:2Adam/m/conv2d/bias
:2Adam/v/conv2d/bias
2:0 2Adam/m/conv3d_1/kernel
2:0 2Adam/v/conv3d_1/kernel
 : 2Adam/m/conv3d_1/bias
 : 2Adam/v/conv3d_1/bias
.:, 2Adam/m/conv2d_1/kernel
.:, 2Adam/v/conv2d_1/kernel
 : 2Adam/m/conv2d_1/bias
 : 2Adam/v/conv2d_1/bias
2:0  2Adam/m/conv3d_2/kernel
2:0  2Adam/v/conv3d_2/kernel
 : 2Adam/m/conv3d_2/bias
 : 2Adam/v/conv3d_2/bias
.:, @2Adam/m/conv2d_2/kernel
.:, @2Adam/v/conv2d_2/kernel
 :@2Adam/m/conv2d_2/bias
 :@2Adam/v/conv2d_2/bias
':%
АА2Adam/m/dense_1/kernel
':%
АА2Adam/v/dense_1/kernel
 :А2Adam/m/dense_1/bias
 :А2Adam/v/dense_1/bias
':%
АА2Adam/m/dense_2/kernel
':%
АА2Adam/v/dense_2/kernel
 :А2Adam/m/dense_2/bias
 :А2Adam/v/dense_2/bias
':%
АА2Adam/m/dense_3/kernel
':%
АА2Adam/v/dense_3/kernel
 :А2Adam/m/dense_3/bias
 :А2Adam/v/dense_3/bias
$:"	А2Adam/m/dense/kernel
$:"	А2Adam/v/dense/kernel
:2Adam/m/dense/bias
:2Adam/v/dense/bias
0
С0
Т1"
trackable_list_wrapper
.
П	variables"
_generic_user_object
:  (2total
:  (2count
0
Х0
Ц1"
trackable_list_wrapper
.
У	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperр
 __inference__wrapped_model_11416╗./78LMUVjkstХЦде│┤╗╝lвi
bв_
]ЪZ
)К&
input_2         pp
-К*
input_1         
0@
к "-к*
(
denseК
dense         я
F__inference_concatenate_layer_call_and_return_conditional_losses_12244дkвh
aв^
\ЪY
*К'
inputs_0         @
+К(
inputs_1         └
к "5в2
+К(
tensor_0         А
Ъ ╔
+__inference_concatenate_layer_call_fn_12237Щkвh
aв^
\ЪY
*К'
inputs_0         @
+К(
inputs_1         └
к "*К'
unknown         А║
C__inference_conv2d_1_layer_call_and_return_conditional_losses_12151sUV7в4
-в*
(К%
inputs         $$
к "4в1
*К'
tensor_0            
Ъ Ф
(__inference_conv2d_1_layer_call_fn_12140hUV7в4
-в*
(К%
inputs         $$
к ")К&
unknown            ║
C__inference_conv2d_2_layer_call_and_return_conditional_losses_12211sst7в4
-в*
(К%
inputs         

 
к "4в1
*К'
tensor_0         @
Ъ Ф
(__inference_conv2d_2_layer_call_fn_12200hst7в4
-в*
(К%
inputs         

 
к ")К&
unknown         @╕
A__inference_conv2d_layer_call_and_return_conditional_losses_12091s787в4
-в*
(К%
inputs         pp
к "4в1
*К'
tensor_0         ll
Ъ Т
&__inference_conv2d_layer_call_fn_12080h787в4
-в*
(К%
inputs         pp
к ")К&
unknown         ll┬
C__inference_conv3d_1_layer_call_and_return_conditional_losses_12131{LM;в8
1в.
,К)
inputs         >
к "8в5
.К+
tensor_0         < 
Ъ Ь
(__inference_conv3d_1_layer_call_fn_12120pLM;в8
1в.
,К)
inputs         >
к "-К*
unknown         < ┬
C__inference_conv3d_2_layer_call_and_return_conditional_losses_12191{jk;в8
1в.
,К)
inputs         
 
к "8в5
.К+
tensor_0          
Ъ Ь
(__inference_conv3d_2_layer_call_fn_12180pjk;в8
1в.
,К)
inputs         
 
к "-К*
unknown          └
A__inference_conv3d_layer_call_and_return_conditional_losses_12071{./;в8
1в.
,К)
inputs         
0@
к "8в5
.К+
tensor_0         .>
Ъ Ъ
&__inference_conv3d_layer_call_fn_12060p./;в8
1в.
,К)
inputs         
0@
к "-К*
unknown         .>н
B__inference_dense_1_layer_call_and_return_conditional_losses_12275gХЦ0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
'__inference_dense_1_layer_call_fn_12264\ХЦ0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
B__inference_dense_2_layer_call_and_return_conditional_losses_12322gде0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
'__inference_dense_2_layer_call_fn_12311\де0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
B__inference_dense_3_layer_call_and_return_conditional_losses_12369g│┤0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
'__inference_dense_3_layer_call_fn_12358\│┤0в-
&в#
!К
inputs         А
к ""К
unknown         Ак
@__inference_dense_layer_call_and_return_conditional_losses_12389f╗╝0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Д
%__inference_dense_layer_call_fn_12378[╗╝0в-
&в#
!К
inputs         А
к "!К
unknown         н
D__inference_dropout_1_layer_call_and_return_conditional_losses_12344e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_1_layer_call_and_return_conditional_losses_12349e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_1_layer_call_fn_12327Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_1_layer_call_fn_12332Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ал
B__inference_dropout_layer_call_and_return_conditional_losses_12297e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ л
B__inference_dropout_layer_call_and_return_conditional_losses_12302e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ Е
'__inference_dropout_layer_call_fn_12280Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЕ
'__inference_dropout_layer_call_fn_12285Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ап
B__inference_flatten_layer_call_and_return_conditional_losses_12255i8в5
.в+
)К&
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Й
'__inference_flatten_layer_call_fn_12249^8в5
.в+
)К&
inputs         А
к ""К
unknown         А╢
C__inference_identity_layer_call_and_return_conditional_losses_12051o7в4
-в*
(К%
inputs         pp
к "4в1
*К'
tensor_0         pp
Ъ Р
(__inference_identity_layer_call_fn_12046d7в4
-в*
(К%
inputs         pp
к ")К&
unknown         ppЇ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_12171еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_1_layer_call_fn_12166ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_12231еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_2_layer_call_fn_12226ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Є
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12111еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╠
-__inference_max_pooling2d_layer_call_fn_12106ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    О
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_12161┐_в\
UвR
PКM
inputsA                                             
к "\вY
RКO
tensor_0A                                             
Ъ ш
/__inference_max_pooling3d_1_layer_call_fn_12156┤_в\
UвR
PКM
inputsA                                             
к "QКN
unknownA                                             О
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_12221┐_в\
UвR
PКM
inputsA                                             
к "\вY
RКO
tensor_0A                                             
Ъ ш
/__inference_max_pooling3d_2_layer_call_fn_12216┤_в\
UвR
PКM
inputsA                                             
к "QКN
unknownA                                             М
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_12101┐_в\
UвR
PКM
inputsA                                             
к "\вY
RКO
tensor_0A                                             
Ъ ц
-__inference_max_pooling3d_layer_call_fn_12096┤_в\
UвR
PКM
inputsA                                             
к "QКN
unknownA                                             З
@__inference_model_layer_call_and_return_conditional_losses_11696┬./78LMUVjkstХЦде│┤╗╝tвq
jвg
]ЪZ
)К&
input_2         pp
-К*
input_1         
0@
p

 
к ",в)
"К
tensor_0         
Ъ З
@__inference_model_layer_call_and_return_conditional_losses_11774┬./78LMUVjkstХЦде│┤╗╝tвq
jвg
]ЪZ
)К&
input_2         pp
-К*
input_1         
0@
p 

 
к ",в)
"К
tensor_0         
Ъ с
%__inference_model_layer_call_fn_11820╖./78LMUVjkstХЦде│┤╗╝tвq
jвg
]ЪZ
)К&
input_2         pp
-К*
input_1         
0@
p

 
к "!К
unknown         с
%__inference_model_layer_call_fn_11866╖./78LMUVjkstХЦде│┤╗╝tвq
jвg
]ЪZ
)К&
input_2         pp
-К*
input_1         
0@
p 

 
к "!К
unknown         Ї
#__inference_signature_wrapper_12041╠./78LMUVjkstХЦде│┤╗╝}вz
в 
sкp
8
input_1-К*
input_1         
0@
4
input_2)К&
input_2         pp"-к*
(
denseК
dense         