Ты
┬С
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
 И"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8╣П
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
Adam/v/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_9/bias
w
'Adam/v/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_9/bias
w
'Adam/m/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/v/dense_9/kernel
А
)Adam/v/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/kernel*
_output_shapes
:	А*
dtype0
З
Adam/m/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/m/dense_9/kernel
А
)Adam/m/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/kernel*
_output_shapes
:	А*
dtype0
Б
Adam/v/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/dense_11/bias
z
(Adam/v/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/dense_11/bias
z
(Adam/m/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/bias*
_output_shapes	
:А*
dtype0
К
Adam/v/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/v/dense_11/kernel
Г
*Adam/v/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/kernel* 
_output_shapes
:
АА*
dtype0
К
Adam/m/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/m/dense_11/kernel
Г
*Adam/m/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/kernel* 
_output_shapes
:
АА*
dtype0
Б
Adam/v/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/dense_10/bias
z
(Adam/v/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/dense_10/bias
z
(Adam/m/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/bias*
_output_shapes	
:А*
dtype0
К
Adam/v/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/v/dense_10/kernel
Г
*Adam/v/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/kernel* 
_output_shapes
:
АА*
dtype0
К
Adam/m/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/m/dense_10/kernel
Г
*Adam/m/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/kernel* 
_output_shapes
:
АА*
dtype0
Г
Adam/v/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/v/conv2d_11/bias
|
)Adam/v/conv2d_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_11/bias*
_output_shapes	
:А*
dtype0
Г
Adam/m/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/m/conv2d_11/bias
|
)Adam/m/conv2d_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_11/bias*
_output_shapes	
:А*
dtype0
Ф
Adam/v/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/v/conv2d_11/kernel
Н
+Adam/v/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_11/kernel*(
_output_shapes
:АА*
dtype0
Ф
Adam/m/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/m/conv2d_11/kernel
Н
+Adam/m/conv2d_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_11/kernel*(
_output_shapes
:АА*
dtype0
Г
Adam/v/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/v/conv2d_10/bias
|
)Adam/v/conv2d_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_10/bias*
_output_shapes	
:А*
dtype0
Г
Adam/m/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/m/conv2d_10/bias
|
)Adam/m/conv2d_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_10/bias*
_output_shapes	
:А*
dtype0
У
Adam/v/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/v/conv2d_10/kernel
М
+Adam/v/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_10/kernel*'
_output_shapes
:@А*
dtype0
У
Adam/m/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/m/conv2d_10/kernel
М
+Adam/m/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_10/kernel*'
_output_shapes
:@А*
dtype0
А
Adam/v/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_9/bias
y
(Adam/v/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_9/bias
y
(Adam/m/conv2d_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_9/kernel
Й
*Adam/v/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_9/kernel*&
_output_shapes
:@*
dtype0
Р
Adam/m/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_9/kernel
Й
*Adam/m/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_9/kernel*&
_output_shapes
:@*
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
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	А*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes	
:А*
dtype0
|
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_11/kernel
u
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:А*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
АА*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:А*
dtype0
Е
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameconv2d_10/kernel
~
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*'
_output_shapes
:@А*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:@*
dtype0
В
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:@*
dtype0
К
serving_default_input_4Placeholder*/
_output_shapes
:         pp*
dtype0*$
shape:         pp
Л
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_36821

NoOpNoOp
Їc
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*пc
valueеcBвc BЫc
Ж
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
╚
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
О
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
╚
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
 <_jit_compiled_convolution_op*
О
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
О
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
ж
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias*
е
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator* 
ж
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias*
е
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
f_random_generator* 
ж
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias*
Z
0
1
+2
,3
:4
;5
O6
P7
^8
_9
m10
n11*
Z
0
1
+2
,3
:4
;5
O6
P7
^8
_9
m10
n11*
* 
░
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ttrace_0
utrace_1* 

vtrace_0
wtrace_1* 
* 
Б
x
_variables
y_iterations
z_learning_rate
{_index_dict
|
_momentums
}_velocities
~_update_step_xla*

serving_default* 

0
1*

0
1*
* 
Ш
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
_Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

Мtrace_0* 

Нtrace_0* 

+0
,1*

+0
,1*
* 
Ш
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 

:0
;1*

:0
;1*
* 
Ш
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

иtrace_0* 

йtrace_0* 
* 
* 
* 
Ц
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

пtrace_0* 

░trace_0* 

O0
P1*

O0
P1*
* 
Ш
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

╢trace_0* 

╖trace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

╜trace_0
╛trace_1* 

┐trace_0
└trace_1* 
* 

^0
_1*

^0
_1*
* 
Ш
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

╞trace_0* 

╟trace_0* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 

═trace_0
╬trace_1* 

╧trace_0
╨trace_1* 
* 

m0
n1*

m0
n1*
* 
Ш
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

╓trace_0* 

╫trace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
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
11*

╪0
┘1*
* 
* 
* 
* 
* 
* 
┌
y0
┌1
█2
▄3
▌4
▐5
▀6
р7
с8
т9
у10
ф11
х12
ц13
ч14
ш15
щ16
ъ17
ы18
ь19
э20
ю21
я22
Ё23
ё24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
┌0
▄1
▐2
р3
т4
ф5
ц6
ш7
ъ8
ь9
ю10
Ё11*
f
█0
▌1
▀2
с3
у4
х5
ч6
щ7
ы8
э9
я10
ё11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
Є	variables
є	keras_api

Їtotal

їcount*
M
Ў	variables
ў	keras_api

°total

∙count
·
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/conv2d_9/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_9/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_9/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_9/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_10/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_10/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_10/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_10/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_11/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_11/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_11/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_11/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_10/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_10/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_10/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_10/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_11/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_11/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_11/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_11/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_9/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_9/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*

Ї0
ї1*

Є	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

°0
∙1*

Ў	variables*
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
ь
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_9/kerneldense_9/bias	iterationlearning_rateAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/biasAdam/m/conv2d_10/kernelAdam/v/conv2d_10/kernelAdam/m/conv2d_10/biasAdam/v/conv2d_10/biasAdam/m/conv2d_11/kernelAdam/v/conv2d_11/kernelAdam/m/conv2d_11/biasAdam/v/conv2d_11/biasAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/biasAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biastotal_1count_1totalcountConst*7
Tin0
.2,*
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
__inference__traced_save_37755
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_9/kerneldense_9/bias	iterationlearning_rateAdam/m/conv2d_9/kernelAdam/v/conv2d_9/kernelAdam/m/conv2d_9/biasAdam/v/conv2d_9/biasAdam/m/conv2d_10/kernelAdam/v/conv2d_10/kernelAdam/m/conv2d_10/biasAdam/v/conv2d_10/biasAdam/m/conv2d_11/kernelAdam/v/conv2d_11/kernelAdam/m/conv2d_11/biasAdam/v/conv2d_11/biasAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/biasAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biastotal_1count_1totalcount*6
Tin/
-2+*
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
!__inference__traced_restore_37890╒┴
╒
═
,__inference_sequential_3_layer_call_fn_36681
input_4!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:	А

unknown_10:
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_36602o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         pp: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name36677:%!

_user_specified_name36675:%
!

_user_specified_name36673:%	!

_user_specified_name36671:%!

_user_specified_name36669:%!

_user_specified_name36667:%!

_user_specified_name36665:%!

_user_specified_name36663:%!

_user_specified_name36661:%!

_user_specified_name36659:%!

_user_specified_name36657:%!

_user_specified_name36655:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_4
щ┴
д
!__inference__traced_restore_37890
file_prefix:
 assignvariableop_conv2d_9_kernel:@.
 assignvariableop_1_conv2d_9_bias:@>
#assignvariableop_2_conv2d_10_kernel:@А0
!assignvariableop_3_conv2d_10_bias:	А?
#assignvariableop_4_conv2d_11_kernel:АА0
!assignvariableop_5_conv2d_11_bias:	А6
"assignvariableop_6_dense_10_kernel:
АА/
 assignvariableop_7_dense_10_bias:	А6
"assignvariableop_8_dense_11_kernel:
АА/
 assignvariableop_9_dense_11_bias:	А5
"assignvariableop_10_dense_9_kernel:	А.
 assignvariableop_11_dense_9_bias:'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: D
*assignvariableop_14_adam_m_conv2d_9_kernel:@D
*assignvariableop_15_adam_v_conv2d_9_kernel:@6
(assignvariableop_16_adam_m_conv2d_9_bias:@6
(assignvariableop_17_adam_v_conv2d_9_bias:@F
+assignvariableop_18_adam_m_conv2d_10_kernel:@АF
+assignvariableop_19_adam_v_conv2d_10_kernel:@А8
)assignvariableop_20_adam_m_conv2d_10_bias:	А8
)assignvariableop_21_adam_v_conv2d_10_bias:	АG
+assignvariableop_22_adam_m_conv2d_11_kernel:ААG
+assignvariableop_23_adam_v_conv2d_11_kernel:АА8
)assignvariableop_24_adam_m_conv2d_11_bias:	А8
)assignvariableop_25_adam_v_conv2d_11_bias:	А>
*assignvariableop_26_adam_m_dense_10_kernel:
АА>
*assignvariableop_27_adam_v_dense_10_kernel:
АА7
(assignvariableop_28_adam_m_dense_10_bias:	А7
(assignvariableop_29_adam_v_dense_10_bias:	А>
*assignvariableop_30_adam_m_dense_11_kernel:
АА>
*assignvariableop_31_adam_v_dense_11_kernel:
АА7
(assignvariableop_32_adam_m_dense_11_bias:	А7
(assignvariableop_33_adam_v_dense_11_bias:	А<
)assignvariableop_34_adam_m_dense_9_kernel:	А<
)assignvariableop_35_adam_v_dense_9_kernel:	А5
'assignvariableop_36_adam_m_dense_9_bias:5
'assignvariableop_37_adam_v_dense_9_bias:%
assignvariableop_38_total_1: %
assignvariableop_39_count_1: #
assignvariableop_40_total: #
assignvariableop_41_count: 
identity_43ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╖
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*▌
value╙B╨+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_conv2d_9_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_9_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_10_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_10_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_11_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_11_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_10_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_10_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_11_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_11_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_9_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_9_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_m_conv2d_9_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_v_conv2d_9_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_m_conv2d_9_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_v_conv2d_9_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_m_conv2d_10_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_v_conv2d_10_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_m_conv2d_10_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_v_conv2d_10_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_m_conv2d_11_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_v_conv2d_11_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_conv2d_11_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_conv2d_11_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_10_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_10_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_dense_10_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_dense_10_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_dense_11_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_dense_11_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_dense_11_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_dense_11_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_dense_9_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_dense_9_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_dense_9_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_dense_9_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ы
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: ┤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_43Identity_43:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%*!

_user_specified_namecount:%)!

_user_specified_nametotal:'(#
!
_user_specified_name	count_1:''#
!
_user_specified_name	total_1:3&/
-
_user_specified_nameAdam/v/dense_9/bias:3%/
-
_user_specified_nameAdam/m/dense_9/bias:5$1
/
_user_specified_nameAdam/v/dense_9/kernel:5#1
/
_user_specified_nameAdam/m/dense_9/kernel:4"0
.
_user_specified_nameAdam/v/dense_11/bias:4!0
.
_user_specified_nameAdam/m/dense_11/bias:6 2
0
_user_specified_nameAdam/v/dense_11/kernel:62
0
_user_specified_nameAdam/m/dense_11/kernel:40
.
_user_specified_nameAdam/v/dense_10/bias:40
.
_user_specified_nameAdam/m/dense_10/bias:62
0
_user_specified_nameAdam/v/dense_10/kernel:62
0
_user_specified_nameAdam/m/dense_10/kernel:51
/
_user_specified_nameAdam/v/conv2d_11/bias:51
/
_user_specified_nameAdam/m/conv2d_11/bias:73
1
_user_specified_nameAdam/v/conv2d_11/kernel:73
1
_user_specified_nameAdam/m/conv2d_11/kernel:51
/
_user_specified_nameAdam/v/conv2d_10/bias:51
/
_user_specified_nameAdam/m/conv2d_10/bias:73
1
_user_specified_nameAdam/v/conv2d_10/kernel:73
1
_user_specified_nameAdam/m/conv2d_10/kernel:40
.
_user_specified_nameAdam/v/conv2d_9/bias:40
.
_user_specified_nameAdam/m/conv2d_9/bias:62
0
_user_specified_nameAdam/v/conv2d_9/kernel:62
0
_user_specified_nameAdam/m/conv2d_9/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_9/bias:.*
(
_user_specified_namedense_9/kernel:-
)
'
_user_specified_namedense_11/bias:/	+
)
_user_specified_namedense_11/kernel:-)
'
_user_specified_namedense_10/bias:/+
)
_user_specified_namedense_10/kernel:.*
(
_user_specified_nameconv2d_11/bias:0,
*
_user_specified_nameconv2d_11/kernel:.*
(
_user_specified_nameconv2d_10/bias:0,
*
_user_specified_nameconv2d_10/kernel:-)
'
_user_specified_nameconv2d_9/bias:/+
)
_user_specified_nameconv2d_9/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
█
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_37056

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧
b
)__inference_dropout_7_layer_call_fn_37034

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_36583p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
я
Э
"__inference_internal_grad_fn_37352
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
█
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_36633

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧
b
)__inference_dropout_6_layer_call_fn_36979

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_36546p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36411

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
Р┴
╬&
__inference__traced_save_37755
file_prefix@
&read_disablecopyonread_conv2d_9_kernel:@4
&read_1_disablecopyonread_conv2d_9_bias:@D
)read_2_disablecopyonread_conv2d_10_kernel:@А6
'read_3_disablecopyonread_conv2d_10_bias:	АE
)read_4_disablecopyonread_conv2d_11_kernel:АА6
'read_5_disablecopyonread_conv2d_11_bias:	А<
(read_6_disablecopyonread_dense_10_kernel:
АА5
&read_7_disablecopyonread_dense_10_bias:	А<
(read_8_disablecopyonread_dense_11_kernel:
АА5
&read_9_disablecopyonread_dense_11_bias:	А;
(read_10_disablecopyonread_dense_9_kernel:	А4
&read_11_disablecopyonread_dense_9_bias:-
#read_12_disablecopyonread_iteration:	 1
'read_13_disablecopyonread_learning_rate: J
0read_14_disablecopyonread_adam_m_conv2d_9_kernel:@J
0read_15_disablecopyonread_adam_v_conv2d_9_kernel:@<
.read_16_disablecopyonread_adam_m_conv2d_9_bias:@<
.read_17_disablecopyonread_adam_v_conv2d_9_bias:@L
1read_18_disablecopyonread_adam_m_conv2d_10_kernel:@АL
1read_19_disablecopyonread_adam_v_conv2d_10_kernel:@А>
/read_20_disablecopyonread_adam_m_conv2d_10_bias:	А>
/read_21_disablecopyonread_adam_v_conv2d_10_bias:	АM
1read_22_disablecopyonread_adam_m_conv2d_11_kernel:ААM
1read_23_disablecopyonread_adam_v_conv2d_11_kernel:АА>
/read_24_disablecopyonread_adam_m_conv2d_11_bias:	А>
/read_25_disablecopyonread_adam_v_conv2d_11_bias:	АD
0read_26_disablecopyonread_adam_m_dense_10_kernel:
ААD
0read_27_disablecopyonread_adam_v_dense_10_kernel:
АА=
.read_28_disablecopyonread_adam_m_dense_10_bias:	А=
.read_29_disablecopyonread_adam_v_dense_10_bias:	АD
0read_30_disablecopyonread_adam_m_dense_11_kernel:
ААD
0read_31_disablecopyonread_adam_v_dense_11_kernel:
АА=
.read_32_disablecopyonread_adam_m_dense_11_bias:	А=
.read_33_disablecopyonread_adam_v_dense_11_bias:	АB
/read_34_disablecopyonread_adam_m_dense_9_kernel:	АB
/read_35_disablecopyonread_adam_v_dense_9_kernel:	А;
-read_36_disablecopyonread_adam_m_dense_9_bias:;
-read_37_disablecopyonread_adam_v_dense_9_bias:+
!read_38_disablecopyonread_total_1: +
!read_39_disablecopyonread_count_1: )
read_40_disablecopyonread_total: )
read_41_disablecopyonread_count: 
savev2_const
identity_85ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_31/DisableCopyOnReadвRead_31/ReadVariableOpвRead_32/DisableCopyOnReadвRead_32/ReadVariableOpвRead_33/DisableCopyOnReadвRead_33/ReadVariableOpвRead_34/DisableCopyOnReadвRead_34/ReadVariableOpвRead_35/DisableCopyOnReadвRead_35/ReadVariableOpвRead_36/DisableCopyOnReadвRead_36/ReadVariableOpвRead_37/DisableCopyOnReadвRead_37/ReadVariableOpвRead_38/DisableCopyOnReadвRead_38/ReadVariableOpвRead_39/DisableCopyOnReadвRead_39/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_40/DisableCopyOnReadвRead_40/ReadVariableOpвRead_41/DisableCopyOnReadвRead_41/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 к
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_9_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_9_bias"/device:CPU:0*
_output_shapes
 в
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_9_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:@}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_conv2d_10_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0v

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аl

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*'
_output_shapes
:@А{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_conv2d_10_bias"/device:CPU:0*
_output_shapes
 д
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_conv2d_10_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 │
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_conv2d_11_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0w

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААm

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*(
_output_shapes
:АА{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_conv2d_11_bias"/device:CPU:0*
_output_shapes
 д
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_conv2d_11_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:А|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 к
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_10_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААz
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 г
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_10_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:А|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_dense_11_kernel"/device:CPU:0*
_output_shapes
 к
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_dense_11_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААz
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_dense_11_bias"/device:CPU:0*
_output_shapes
 г
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_dense_11_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
:А}
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 л
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_dense_9_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	А{
Read_11/DisableCopyOnReadDisableCopyOnRead&read_11_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 д
Read_11/ReadVariableOpReadVariableOp&read_11_disablecopyonread_dense_9_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_iteration^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 б
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_learning_rate^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_adam_m_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 ║
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_adam_m_conv2d_9_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Е
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_v_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 ║
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_v_conv2d_9_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
:@Г
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_m_conv2d_9_bias"/device:CPU:0*
_output_shapes
 м
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_m_conv2d_9_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_adam_v_conv2d_9_bias"/device:CPU:0*
_output_shapes
 м
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_adam_v_conv2d_9_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ж
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_adam_m_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_adam_m_conv2d_10_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*'
_output_shapes
:@АЖ
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_adam_v_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_adam_v_conv2d_10_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@А*
dtype0x
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@Аn
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*'
_output_shapes
:@АД
Read_20/DisableCopyOnReadDisableCopyOnRead/read_20_disablecopyonread_adam_m_conv2d_10_bias"/device:CPU:0*
_output_shapes
 о
Read_20/ReadVariableOpReadVariableOp/read_20_disablecopyonread_adam_m_conv2d_10_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_v_conv2d_10_bias"/device:CPU:0*
_output_shapes
 о
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_v_conv2d_10_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЖ
Read_22/DisableCopyOnReadDisableCopyOnRead1read_22_disablecopyonread_adam_m_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 ╜
Read_22/ReadVariableOpReadVariableOp1read_22_disablecopyonread_adam_m_conv2d_11_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0y
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААЖ
Read_23/DisableCopyOnReadDisableCopyOnRead1read_23_disablecopyonread_adam_v_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 ╜
Read_23/ReadVariableOpReadVariableOp1read_23_disablecopyonread_adam_v_conv2d_11_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0y
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААД
Read_24/DisableCopyOnReadDisableCopyOnRead/read_24_disablecopyonread_adam_m_conv2d_11_bias"/device:CPU:0*
_output_shapes
 о
Read_24/ReadVariableOpReadVariableOp/read_24_disablecopyonread_adam_m_conv2d_11_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_adam_v_conv2d_11_bias"/device:CPU:0*
_output_shapes
 о
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_adam_v_conv2d_11_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_10_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_10_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЕ
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_10_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_10_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААГ
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_dense_10_bias"/device:CPU:0*
_output_shapes
 н
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_dense_10_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_dense_10_bias"/device:CPU:0*
_output_shapes
 н
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_dense_10_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_dense_11_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_dense_11_kernel^Read_30/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЕ
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_dense_11_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_dense_11_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААГ
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_dense_11_bias"/device:CPU:0*
_output_shapes
 н
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_dense_11_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_dense_11_bias"/device:CPU:0*
_output_shapes
 н
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_dense_11_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_34/DisableCopyOnReadDisableCopyOnRead/read_34_disablecopyonread_adam_m_dense_9_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_34/ReadVariableOpReadVariableOp/read_34_disablecopyonread_adam_m_dense_9_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	АД
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_v_dense_9_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_v_dense_9_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_36/DisableCopyOnReadDisableCopyOnRead-read_36_disablecopyonread_adam_m_dense_9_bias"/device:CPU:0*
_output_shapes
 л
Read_36/ReadVariableOpReadVariableOp-read_36_disablecopyonread_adam_m_dense_9_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_37/DisableCopyOnReadDisableCopyOnRead-read_37_disablecopyonread_adam_v_dense_9_bias"/device:CPU:0*
_output_shapes
 л
Read_37/ReadVariableOpReadVariableOp-read_37_disablecopyonread_adam_v_dense_9_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_38/DisableCopyOnReadDisableCopyOnRead!read_38_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_38/ReadVariableOpReadVariableOp!read_38_disablecopyonread_total_1^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_39/DisableCopyOnReadDisableCopyOnRead!read_39_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_39/ReadVariableOpReadVariableOp!read_39_disablecopyonread_count_1^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_40/DisableCopyOnReadDisableCopyOnReadread_40_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_40/ReadVariableOpReadVariableOpread_40_disablecopyonread_total^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_41/DisableCopyOnReadDisableCopyOnReadread_41_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_41/ReadVariableOpReadVariableOpread_41_disablecopyonread_count^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: ┤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*▌
value╙B╨+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH├
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Щ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *9
dtypes/
-2+	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_84Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_85IdentityIdentity_84:output:0^NoOp*
T0*
_output_shapes
: ╤
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_85Identity_85:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_41/ReadVariableOpRead_41/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=+9

_output_shapes
: 

_user_specified_nameConst:%*!

_user_specified_namecount:%)!

_user_specified_nametotal:'(#
!
_user_specified_name	count_1:''#
!
_user_specified_name	total_1:3&/
-
_user_specified_nameAdam/v/dense_9/bias:3%/
-
_user_specified_nameAdam/m/dense_9/bias:5$1
/
_user_specified_nameAdam/v/dense_9/kernel:5#1
/
_user_specified_nameAdam/m/dense_9/kernel:4"0
.
_user_specified_nameAdam/v/dense_11/bias:4!0
.
_user_specified_nameAdam/m/dense_11/bias:6 2
0
_user_specified_nameAdam/v/dense_11/kernel:62
0
_user_specified_nameAdam/m/dense_11/kernel:40
.
_user_specified_nameAdam/v/dense_10/bias:40
.
_user_specified_nameAdam/m/dense_10/bias:62
0
_user_specified_nameAdam/v/dense_10/kernel:62
0
_user_specified_nameAdam/m/dense_10/kernel:51
/
_user_specified_nameAdam/v/conv2d_11/bias:51
/
_user_specified_nameAdam/m/conv2d_11/bias:73
1
_user_specified_nameAdam/v/conv2d_11/kernel:73
1
_user_specified_nameAdam/m/conv2d_11/kernel:51
/
_user_specified_nameAdam/v/conv2d_10/bias:51
/
_user_specified_nameAdam/m/conv2d_10/bias:73
1
_user_specified_nameAdam/v/conv2d_10/kernel:73
1
_user_specified_nameAdam/m/conv2d_10/kernel:40
.
_user_specified_nameAdam/v/conv2d_9/bias:40
.
_user_specified_nameAdam/m/conv2d_9/bias:62
0
_user_specified_nameAdam/v/conv2d_9/kernel:62
0
_user_specified_nameAdam/m/conv2d_9/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_9/bias:.*
(
_user_specified_namedense_9/kernel:-
)
'
_user_specified_namedense_11/bias:/	+
)
_user_specified_namedense_11/kernel:-)
'
_user_specified_namedense_10/bias:/+
)
_user_specified_namedense_10/kernel:.*
(
_user_specified_nameconv2d_11/bias:0,
*
_user_specified_nameconv2d_11/kernel:.*
(
_user_specified_nameconv2d_10/bias:0,
*
_user_specified_nameconv2d_10/kernel:-)
'
_user_specified_nameconv2d_9/bias:/+
)
_user_specified_nameconv2d_9/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╞
Э
"__inference_internal_grad_fn_37217
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
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         АV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:         АU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:         АP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:         А[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:         АW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:         АV
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
:         АR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         А: : :         А:QM
(
_output_shapes
:         А
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
:         А
(
_user_specified_nameresult_grads_1:А |
&
 _has_manual_control_dependencies(
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
╚
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_36946

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
М
В
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36497

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
_gradient_op_typeCustomGradient-36488*N
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
╕
L
0__inference_max_pooling2d_10_layer_call_fn_36892

inputs
identity┘
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36411Г
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
Жs
ф
 __inference__wrapped_model_36396
input_4N
4sequential_3_conv2d_9_conv2d_readvariableop_resource:@C
5sequential_3_conv2d_9_biasadd_readvariableop_resource:@P
5sequential_3_conv2d_10_conv2d_readvariableop_resource:@АE
6sequential_3_conv2d_10_biasadd_readvariableop_resource:	АQ
5sequential_3_conv2d_11_conv2d_readvariableop_resource:ААE
6sequential_3_conv2d_11_biasadd_readvariableop_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	АH
4sequential_3_dense_11_matmul_readvariableop_resource:
ААD
5sequential_3_dense_11_biasadd_readvariableop_resource:	АF
3sequential_3_dense_9_matmul_readvariableop_resource:	АB
4sequential_3_dense_9_biasadd_readvariableop_resource:
identityИв-sequential_3/conv2d_10/BiasAdd/ReadVariableOpв,sequential_3/conv2d_10/Conv2D/ReadVariableOpв-sequential_3/conv2d_11/BiasAdd/ReadVariableOpв,sequential_3/conv2d_11/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв,sequential_3/dense_11/BiasAdd/ReadVariableOpв+sequential_3/dense_11/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpи
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╟
sequential_3/conv2d_9/Conv2DConv2Dinput_43sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@*
paddingVALID*
strides
Ю
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┐
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll@_
sequential_3/conv2d_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
sequential_3/conv2d_9/mulMul#sequential_3/conv2d_9/beta:output:0&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         ll@Б
sequential_3/conv2d_9/SigmoidSigmoidsequential_3/conv2d_9/mul:z:0*
T0*/
_output_shapes
:         ll@з
sequential_3/conv2d_9/mul_1Mul&sequential_3/conv2d_9/BiasAdd:output:0!sequential_3/conv2d_9/Sigmoid:y:0*
T0*/
_output_shapes
:         ll@Е
sequential_3/conv2d_9/IdentityIdentitysequential_3/conv2d_9/mul_1:z:0*
T0*/
_output_shapes
:         ll@г
sequential_3/conv2d_9/IdentityN	IdentityNsequential_3/conv2d_9/mul_1:z:0&sequential_3/conv2d_9/BiasAdd:output:0#sequential_3/conv2d_9/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36313*L
_output_shapes:
8:         ll@:         ll@: ╞
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/IdentityN:output:0*/
_output_shapes
:         $$@*
ksize
*
paddingVALID*
strides
л
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ё
sequential_3/conv2d_10/Conv2DConv2D-sequential_3/max_pooling2d_9/MaxPool:output:04sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
б
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А`
sequential_3/conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
sequential_3/conv2d_10/mulMul$sequential_3/conv2d_10/beta:output:0'sequential_3/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:           АД
sequential_3/conv2d_10/SigmoidSigmoidsequential_3/conv2d_10/mul:z:0*
T0*0
_output_shapes
:           Ал
sequential_3/conv2d_10/mul_1Mul'sequential_3/conv2d_10/BiasAdd:output:0"sequential_3/conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:           АИ
sequential_3/conv2d_10/IdentityIdentity sequential_3/conv2d_10/mul_1:z:0*
T0*0
_output_shapes
:           Ай
 sequential_3/conv2d_10/IdentityN	IdentityN sequential_3/conv2d_10/mul_1:z:0'sequential_3/conv2d_10/BiasAdd:output:0$sequential_3/conv2d_10/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36329*N
_output_shapes<
::           А:           А: ╔
%sequential_3/max_pooling2d_10/MaxPoolMaxPool)sequential_3/conv2d_10/IdentityN:output:0*0
_output_shapes
:         

А*
ksize
*
paddingVALID*
strides
м
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ё
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
б
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А`
sequential_3/conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
sequential_3/conv2d_11/mulMul$sequential_3/conv2d_11/beta:output:0'sequential_3/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         АД
sequential_3/conv2d_11/SigmoidSigmoidsequential_3/conv2d_11/mul:z:0*
T0*0
_output_shapes
:         Ал
sequential_3/conv2d_11/mul_1Mul'sequential_3/conv2d_11/BiasAdd:output:0"sequential_3/conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:         АИ
sequential_3/conv2d_11/IdentityIdentity sequential_3/conv2d_11/mul_1:z:0*
T0*0
_output_shapes
:         Ай
 sequential_3/conv2d_11/IdentityN	IdentityN sequential_3/conv2d_11/mul_1:z:0'sequential_3/conv2d_11/BiasAdd:output:0$sequential_3/conv2d_11/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36345*N
_output_shapes<
::         А:         А: ╔
%sequential_3/max_pooling2d_11/MaxPoolMaxPool)sequential_3/conv2d_11/IdentityN:output:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       │
sequential_3/flatten_3/ReshapeReshape.sequential_3/max_pooling2d_11/MaxPool:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:         Ав
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╖
sequential_3/dense_10/MatMulMatMul'sequential_3/flatten_3/Reshape:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А_
sequential_3/dense_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
sequential_3/dense_10/mulMul#sequential_3/dense_10/beta:output:0&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         Аz
sequential_3/dense_10/SigmoidSigmoidsequential_3/dense_10/mul:z:0*
T0*(
_output_shapes
:         Аа
sequential_3/dense_10/mul_1Mul&sequential_3/dense_10/BiasAdd:output:0!sequential_3/dense_10/Sigmoid:y:0*
T0*(
_output_shapes
:         А~
sequential_3/dense_10/IdentityIdentitysequential_3/dense_10/mul_1:z:0*
T0*(
_output_shapes
:         АХ
sequential_3/dense_10/IdentityN	IdentityNsequential_3/dense_10/mul_1:z:0&sequential_3/dense_10/BiasAdd:output:0#sequential_3/dense_10/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36363*>
_output_shapes,
*:         А:         А: И
sequential_3/dropout_6/IdentityIdentity(sequential_3/dense_10/IdentityN:output:0*
T0*(
_output_shapes
:         Ав
+sequential_3/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╕
sequential_3/dense_11/MatMulMatMul(sequential_3/dropout_6/Identity:output:03sequential_3/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_3/dense_11/BiasAddBiasAdd&sequential_3/dense_11/MatMul:product:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А_
sequential_3/dense_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
sequential_3/dense_11/mulMul#sequential_3/dense_11/beta:output:0&sequential_3/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:         Аz
sequential_3/dense_11/SigmoidSigmoidsequential_3/dense_11/mul:z:0*
T0*(
_output_shapes
:         Аа
sequential_3/dense_11/mul_1Mul&sequential_3/dense_11/BiasAdd:output:0!sequential_3/dense_11/Sigmoid:y:0*
T0*(
_output_shapes
:         А~
sequential_3/dense_11/IdentityIdentitysequential_3/dense_11/mul_1:z:0*
T0*(
_output_shapes
:         АХ
sequential_3/dense_11/IdentityN	IdentityNsequential_3/dense_11/mul_1:z:0&sequential_3/dense_11/BiasAdd:output:0#sequential_3/dense_11/beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36379*>
_output_shapes,
*:         А:         А: И
sequential_3/dropout_7/IdentityIdentity(sequential_3/dense_11/IdentityN:output:0*
T0*(
_output_shapes
:         АЯ
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0╡
sequential_3/dense_9/MatMulMatMul(sequential_3/dropout_7/Identity:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
sequential_3/dense_9/SoftmaxSoftmax%sequential_3/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_3/dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╥
NoOpNoOp.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp,^sequential_3/dense_11/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         pp: : : : : : : : : : : : 2^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_11/MatMul/ReadVariableOp+sequential_3/dense_11/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:($
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
:         pp
!
_user_specified_name	input_4
Ы
б
)__inference_conv2d_11_layer_call_fn_36906

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallт
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36497x
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

_user_specified_name36902:%!

_user_specified_name36900:X T
0
_output_shapes
:         

А
 
_user_specified_nameinputs
И
Б
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36887

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
_gradient_op_typeCustomGradient-36878*N
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
И8
и
G__inference_sequential_3_layer_call_and_return_conditional_losses_36602
input_4(
conv2d_9_36448:@
conv2d_9_36450:@*
conv2d_10_36473:@А
conv2d_10_36475:	А+
conv2d_11_36498:АА
conv2d_11_36500:	А"
dense_10_36530:
АА
dense_10_36532:	А"
dense_11_36567:
АА
dense_11_36569:	А 
dense_9_36596:	А
dense_9_36598:
identityИв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallв dense_10/StatefulPartitionedCallв dense_11/StatefulPartitionedCallвdense_9/StatefulPartitionedCallв!dropout_6/StatefulPartitionedCallв!dropout_7/StatefulPartitionedCallЎ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_9_36448conv2d_9_36450*
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36447Ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36401Ь
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0conv2d_10_36473conv2d_10_36475*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36472Ї
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36411Э
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_36498conv2d_11_36500*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36497Ї
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36421▌
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_36509К
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_10_36530dense_10_36532*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_36529э
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_36546Т
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_11_36567dense_11_36569*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_36566С
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_36583Н
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_9_36596dense_9_36598*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_36595w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╜
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         pp: : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:%!

_user_specified_name36598:%!

_user_specified_name36596:%
!

_user_specified_name36569:%	!

_user_specified_name36567:%!

_user_specified_name36532:%!

_user_specified_name36530:%!

_user_specified_name36500:%!

_user_specified_name36498:%!

_user_specified_name36475:%!

_user_specified_name36473:%!

_user_specified_name36450:%!

_user_specified_name36448:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_4
Т
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36859

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
Я

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_36546

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
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Я

c
D__inference_dropout_7_layer_call_and_return_conditional_losses_36583

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
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36401

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
╞
Э
"__inference_internal_grad_fn_37190
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
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         АV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:         АU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:         АP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:         А[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:         АW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:         АV
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
:         АR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         А: : :         А:QM
(
_output_shapes
:         А
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
:         А
(
_user_specified_nameresult_grads_1:А |
&
 _has_manual_control_dependencies(
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
▒
E
)__inference_flatten_3_layer_call_fn_36940

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
D__inference_flatten_3_layer_call_and_return_conditional_losses_36509a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_9_layer_call_fn_36830

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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36447w
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

_user_specified_name36826:%!

_user_specified_name36824:W S
/
_output_shapes
:         pp
 
_user_specified_nameinputs
г
╔
"__inference_internal_grad_fn_37460
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_3_conv2d_9_beta%
!mul_sequential_3_conv2d_9_biasadd
identity

identity_1Ш
mulMulmul_sequential_3_conv2d_9_beta!mul_sequential_3_conv2d_9_biasadd^result_grads_0*
T0*/
_output_shapes
:         ll@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:         ll@Й
mul_1Mulmul_sequential_3_conv2d_9_beta!mul_sequential_3_conv2d_9_biasadd*
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
:         ll@m
SquareSquare!mul_sequential_3_conv2d_9_biasadd*
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
U:         ll@:         ll@: : :         ll@:nj
/
_output_shapes
:         ll@
7
_user_specified_namesequential_3/conv2d_9/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_3/conv2d_9/beta:FB
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
╥

Ї
B__inference_dense_9_layer_call_and_return_conditional_losses_37076

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
:         А: : 20
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
:         А
 
_user_specified_nameinputs
Ф5
р
G__inference_sequential_3_layer_call_and_return_conditional_losses_36652
input_4(
conv2d_9_36605:@
conv2d_9_36607:@*
conv2d_10_36611:@А
conv2d_10_36613:	А+
conv2d_11_36617:АА
conv2d_11_36619:	А"
dense_10_36624:
АА
dense_10_36626:	А"
dense_11_36635:
АА
dense_11_36637:	А 
dense_9_36646:	А
dense_9_36648:
identityИв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallв dense_10/StatefulPartitionedCallв dense_11/StatefulPartitionedCallвdense_9/StatefulPartitionedCallЎ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_9_36605conv2d_9_36607*
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36447Ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36401Ь
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0conv2d_10_36611conv2d_10_36613*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36472Ї
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36411Э
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_36617conv2d_11_36619*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36497Ї
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36421▌
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_36509К
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_10_36624dense_10_36626*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_36529▌
dropout_6/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_36633К
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_11_36635dense_11_36637*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_36566▌
dropout_7/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_36644Е
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_9_36646dense_9_36648*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_36595w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ї
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         pp: : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:%!

_user_specified_name36648:%!

_user_specified_name36646:%
!

_user_specified_name36637:%	!

_user_specified_name36635:%!

_user_specified_name36626:%!

_user_specified_name36624:%!

_user_specified_name36619:%!

_user_specified_name36617:%!

_user_specified_name36613:%!

_user_specified_name36611:%!

_user_specified_name36607:%!

_user_specified_name36605:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_4
И
Б
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36472

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
_gradient_op_typeCustomGradient-36463*N
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
█
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_37001

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36421

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
╥

Ї
B__inference_dense_9_layer_call_and_return_conditional_losses_36595

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
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
:         А: : 20
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
:         А
 
_user_specified_nameinputs
Я

c
D__inference_dropout_7_layer_call_and_return_conditional_losses_37051

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
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36897

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
"__inference_internal_grad_fn_37379
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
У
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36935

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
·
∙
C__inference_dense_11_layer_call_and_return_conditional_losses_37029

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         А^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:         А╜
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-37020*>
_output_shapes,
*:         А:         А: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
└
╦
"__inference_internal_grad_fn_37514
result_grads_0
result_grads_1
result_grads_2#
mul_sequential_3_conv2d_11_beta&
"mul_sequential_3_conv2d_11_biasadd
identity

identity_1Ы
mulMulmul_sequential_3_conv2d_11_beta"mul_sequential_3_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:         АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:         АМ
mul_1Mulmul_sequential_3_conv2d_11_beta"mul_sequential_3_conv2d_11_biasadd*
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
:         Аo
SquareSquare"mul_sequential_3_conv2d_11_biasadd*
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
X:         А:         А: : :         А:pl
0
_output_shapes
:         А
8
_user_specified_name sequential_3/conv2d_11/BiasAdd:SO

_output_shapes
: 
5
_user_specified_namesequential_3/conv2d_11/beta:FB
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
О
╔
"__inference_internal_grad_fn_37568
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_3_dense_11_beta%
!mul_sequential_3_dense_11_biasadd
identity

identity_1С
mulMulmul_sequential_3_dense_11_beta!mul_sequential_3_dense_11_biasadd^result_grads_0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         АВ
mul_1Mulmul_sequential_3_dense_11_beta!mul_sequential_3_dense_11_biasadd*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:         АU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:         Аf
SquareSquare!mul_sequential_3_dense_11_biasadd*
T0*(
_output_shapes
:         А[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:         АW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:         АV
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
:         АR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         А: : :         А:gc
(
_output_shapes
:         А
7
_user_specified_namesequential_3/dense_11/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_3/dense_11/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_1:А |
&
 _has_manual_control_dependencies(
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
╒
═
,__inference_sequential_3_layer_call_fn_36710
input_4!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:	А

unknown_10:
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_36652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         pp: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name36706:%!

_user_specified_name36704:%
!

_user_specified_name36702:%	!

_user_specified_name36700:%!

_user_specified_name36698:%!

_user_specified_name36696:%!

_user_specified_name36694:%!

_user_specified_name36692:%!

_user_specified_name36690:%!

_user_specified_name36688:%!

_user_specified_name36686:%!

_user_specified_name36684:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_4
М
В
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36925

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
_gradient_op_typeCustomGradient-36916*N
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
╞
Э
"__inference_internal_grad_fn_37271
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
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         АV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:         АU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:         АP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:         А[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:         АW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:         АV
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
:         АR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         А: : :         А:QM
(
_output_shapes
:         А
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
:         А
(
_user_specified_nameresult_grads_1:А |
&
 _has_manual_control_dependencies(
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
ы
Х
'__inference_dense_9_layer_call_fn_37065

inputs
unknown:	А
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
B__inference_dense_9_layer_call_and_return_conditional_losses_36595o
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
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name37061:%!

_user_specified_name37059:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╚
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_36509

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
я
Э
"__inference_internal_grad_fn_37298
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
б
E
)__inference_dropout_6_layer_call_fn_36984

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_36633a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·
∙
C__inference_dense_10_layer_call_and_return_conditional_losses_36529

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         А^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:         А╜
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36520*>
_output_shapes,
*:         А:         А: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
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
Ш
а
)__inference_conv2d_10_layer_call_fn_36868

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallт
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36472x
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

_user_specified_name36864:%!

_user_specified_name36862:W S
/
_output_shapes
:         $$@
 
_user_specified_nameinputs
█
Э
"__inference_internal_grad_fn_37433
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
·
■
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36447

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
_gradient_op_typeCustomGradient-36438*L
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
╕
L
0__inference_max_pooling2d_11_layer_call_fn_36930

inputs
identity┘
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36421Г
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
Я

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_36996

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
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
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
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
└
╦
"__inference_internal_grad_fn_37487
result_grads_0
result_grads_1
result_grads_2#
mul_sequential_3_conv2d_10_beta&
"mul_sequential_3_conv2d_10_biasadd
identity

identity_1Ы
mulMulmul_sequential_3_conv2d_10_beta"mul_sequential_3_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:           АV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:           АМ
mul_1Mulmul_sequential_3_conv2d_10_beta"mul_sequential_3_conv2d_10_biasadd*
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
:           Аo
SquareSquare"mul_sequential_3_conv2d_10_biasadd*
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
X:           А:           А: : :           А:pl
0
_output_shapes
:           А
8
_user_specified_name sequential_3/conv2d_10/BiasAdd:SO

_output_shapes
: 
5
_user_specified_namesequential_3/conv2d_10/beta:FB
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
█
Э
"__inference_internal_grad_fn_37406
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
█
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_36644

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
О
╔
"__inference_internal_grad_fn_37541
result_grads_0
result_grads_1
result_grads_2"
mul_sequential_3_dense_10_beta%
!mul_sequential_3_dense_10_biasadd
identity

identity_1С
mulMulmul_sequential_3_dense_10_beta!mul_sequential_3_dense_10_biasadd^result_grads_0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         АВ
mul_1Mulmul_sequential_3_dense_10_beta!mul_sequential_3_dense_10_biasadd*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:         АU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:         Аf
SquareSquare!mul_sequential_3_dense_10_biasadd*
T0*(
_output_shapes
:         А[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:         АW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:         АV
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
:         АR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         А: : :         А:gc
(
_output_shapes
:         А
7
_user_specified_namesequential_3/dense_10/BiasAdd:RN

_output_shapes
: 
4
_user_specified_namesequential_3/dense_10/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_1:А |
&
 _has_manual_control_dependencies(
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
ё
Ш
(__inference_dense_11_layer_call_fn_37010

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_36566p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name37006:%!

_user_specified_name37004:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·
■
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36849

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
_gradient_op_typeCustomGradient-36840*L
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
я
Э
"__inference_internal_grad_fn_37325
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
б
E
)__inference_dropout_7_layer_call_fn_37039

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
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_36644a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·
∙
C__inference_dense_10_layer_call_and_return_conditional_losses_36974

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         А^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:         А╜
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36965*>
_output_shapes,
*:         А:         А: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
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
ё
Ш
(__inference_dense_10_layer_call_fn_36955

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_36529p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name36951:%!

_user_specified_name36949:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·
∙
C__inference_dense_11_layer_call_and_return_conditional_losses_36566

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         А^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:         А╜
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*+
_gradient_op_typeCustomGradient-36557*>
_output_shapes,
*:         А:         А: d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
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
:         А
 
_user_specified_nameinputs
╞
Э
"__inference_internal_grad_fn_37244
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
:         АN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:         АV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:         АJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:         АU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:         АP
SquareSquaremul_biasadd*
T0*(
_output_shapes
:         А[
mul_4Mulresult_grads_0
Square:y:0*
T0*(
_output_shapes
:         АW
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*(
_output_shapes
:         АL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?^
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АU
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*(
_output_shapes
:         АV
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
:         АR
IdentityIdentity	mul_7:z:0*
T0*(
_output_shapes
:         АE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         А: : :         А:QM
(
_output_shapes
:         А
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
:         А
(
_user_specified_nameresult_grads_1:А |
&
 _has_manual_control_dependencies(
(
_output_shapes
:         А
(
_user_specified_nameresult_grads_0
╢
K
/__inference_max_pooling2d_9_layer_call_fn_36854

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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36401Г
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
е
─
#__inference_signature_wrapper_36821
input_4!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:	А

unknown_10:
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_36396o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         pp: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name36817:%!

_user_specified_name36815:%
!

_user_specified_name36813:%	!

_user_specified_name36811:%!

_user_specified_name36809:%!

_user_specified_name36807:%!

_user_specified_name36805:%!

_user_specified_name36803:%!

_user_specified_name36801:%!

_user_specified_name36799:%!

_user_specified_name36797:%!

_user_specified_name36795:X T
/
_output_shapes
:         pp
!
_user_specified_name	input_4:
"__inference_internal_grad_fn_37190CustomGradient-37020:
"__inference_internal_grad_fn_37217CustomGradient-36557:
"__inference_internal_grad_fn_37244CustomGradient-36965:
"__inference_internal_grad_fn_37271CustomGradient-36520:
"__inference_internal_grad_fn_37298CustomGradient-36916:
"__inference_internal_grad_fn_37325CustomGradient-36488:
"__inference_internal_grad_fn_37352CustomGradient-36878:
"__inference_internal_grad_fn_37379CustomGradient-36463:
"__inference_internal_grad_fn_37406CustomGradient-36840:
"__inference_internal_grad_fn_37433CustomGradient-36438:
"__inference_internal_grad_fn_37460CustomGradient-36313:
"__inference_internal_grad_fn_37487CustomGradient-36329:
"__inference_internal_grad_fn_37514CustomGradient-36345:
"__inference_internal_grad_fn_37541CustomGradient-36363:
"__inference_internal_grad_fn_37568CustomGradient-36379"зL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▓
serving_defaultЮ
C
input_48
serving_default_input_4:0         pp;
dense_90
StatefulPartitionedCall:0         tensorflow/serving/predict:и─
а
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
е
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
е
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
 <_jit_compiled_convolution_op"
_tf_keras_layer
е
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
е
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias"
_tf_keras_layer
╝
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator"
_tf_keras_layer
╗
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias"
_tf_keras_layer
╝
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
f_random_generator"
_tf_keras_layer
╗
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias"
_tf_keras_layer
v
0
1
+2
,3
:4
;5
O6
P7
^8
_9
m10
n11"
trackable_list_wrapper
v
0
1
+2
,3
:4
;5
O6
P7
^8
_9
m10
n11"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╦
ttrace_0
utrace_12Ф
,__inference_sequential_3_layer_call_fn_36681
,__inference_sequential_3_layer_call_fn_36710╡
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
 zttrace_0zutrace_1
Б
vtrace_0
wtrace_12╩
G__inference_sequential_3_layer_call_and_return_conditional_losses_36602
G__inference_sequential_3_layer_call_and_return_conditional_losses_36652╡
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
 zvtrace_0zwtrace_1
╦B╚
 __inference__wrapped_model_36396input_4"Ш
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
Ь
x
_variables
y_iterations
z_learning_rate
{_index_dict
|
_momentums
}_velocities
~_update_step_xla"
experimentalOptimizer
,
serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ф
Еtrace_02┼
(__inference_conv2d_9_layer_call_fn_36830Ш
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
 zЕtrace_0
 
Жtrace_02р
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36849Ш
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
 zЖtrace_0
):'@2conv2d_9/kernel
:@2conv2d_9/bias
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
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ы
Мtrace_02╠
/__inference_max_pooling2d_9_layer_call_fn_36854Ш
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
 zМtrace_0
Ж
Нtrace_02ч
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36859Ш
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
 zНtrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
х
Уtrace_02╞
)__inference_conv2d_10_layer_call_fn_36868Ш
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
 zУtrace_0
А
Фtrace_02с
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36887Ш
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
 zФtrace_0
+:)@А2conv2d_10/kernel
:А2conv2d_10/bias
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
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ь
Ъtrace_02═
0__inference_max_pooling2d_10_layer_call_fn_36892Ш
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
 zЪtrace_0
З
Ыtrace_02ш
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36897Ш
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
 zЫtrace_0
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
х
бtrace_02╞
)__inference_conv2d_11_layer_call_fn_36906Ш
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
 zбtrace_0
А
вtrace_02с
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36925Ш
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
 zвtrace_0
,:*АА2conv2d_11/kernel
:А2conv2d_11/bias
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
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
ь
иtrace_02═
0__inference_max_pooling2d_11_layer_call_fn_36930Ш
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
 zиtrace_0
З
йtrace_02ш
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36935Ш
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
 zйtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
х
пtrace_02╞
)__inference_flatten_3_layer_call_fn_36940Ш
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
 zпtrace_0
А
░trace_02с
D__inference_flatten_3_layer_call_and_return_conditional_losses_36946Ш
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
 z░trace_0
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
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ф
╢trace_02┼
(__inference_dense_10_layer_call_fn_36955Ш
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
 z╢trace_0
 
╖trace_02р
C__inference_dense_10_layer_call_and_return_conditional_losses_36974Ш
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
 z╖trace_0
#:!
АА2dense_10/kernel
:А2dense_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
╜
╜trace_0
╛trace_12В
)__inference_dropout_6_layer_call_fn_36979
)__inference_dropout_6_layer_call_fn_36984й
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
 z╜trace_0z╛trace_1
є
┐trace_0
└trace_12╕
D__inference_dropout_6_layer_call_and_return_conditional_losses_36996
D__inference_dropout_6_layer_call_and_return_conditional_losses_37001й
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
 z┐trace_0z└trace_1
"
_generic_user_object
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
ф
╞trace_02┼
(__inference_dense_11_layer_call_fn_37010Ш
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
 z╞trace_0
 
╟trace_02р
C__inference_dense_11_layer_call_and_return_conditional_losses_37029Ш
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
 z╟trace_0
#:!
АА2dense_11/kernel
:А2dense_11/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
╜
═trace_0
╬trace_12В
)__inference_dropout_7_layer_call_fn_37034
)__inference_dropout_7_layer_call_fn_37039й
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
 z═trace_0z╬trace_1
є
╧trace_0
╨trace_12╕
D__inference_dropout_7_layer_call_and_return_conditional_losses_37051
D__inference_dropout_7_layer_call_and_return_conditional_losses_37056й
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
 z╧trace_0z╨trace_1
"
_generic_user_object
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
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
у
╓trace_02─
'__inference_dense_9_layer_call_fn_37065Ш
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
 z╓trace_0
■
╫trace_02▀
B__inference_dense_9_layer_call_and_return_conditional_losses_37076Ш
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
 z╫trace_0
!:	А2dense_9/kernel
:2dense_9/bias
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
0
╪0
┘1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
,__inference_sequential_3_layer_call_fn_36681input_4"м
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
ыBш
,__inference_sequential_3_layer_call_fn_36710input_4"м
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
ЖBГ
G__inference_sequential_3_layer_call_and_return_conditional_losses_36602input_4"м
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
ЖBГ
G__inference_sequential_3_layer_call_and_return_conditional_losses_36652input_4"м
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
Ў
y0
┌1
█2
▄3
▌4
▐5
▀6
р7
с8
т9
у10
ф11
х12
ц13
ч14
ш15
щ16
ъ17
ы18
ь19
э20
ю21
я22
Ё23
ё24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
В
┌0
▄1
▐2
р3
т4
ф5
ц6
ш7
ъ8
ь9
ю10
Ё11"
trackable_list_wrapper
В
█0
▌1
▀2
с3
у4
х5
ч6
щ7
ы8
э9
я10
ё11"
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
╧B╠
#__inference_signature_wrapper_36821input_4"Щ
Т▓О
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ
	jinput_4
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
(__inference_conv2d_9_layer_call_fn_36830inputs"Ш
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36849inputs"Ш
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
/__inference_max_pooling2d_9_layer_call_fn_36854inputs"Ш
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36859inputs"Ш
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
)__inference_conv2d_10_layer_call_fn_36868inputs"Ш
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
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36887inputs"Ш
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
┌B╫
0__inference_max_pooling2d_10_layer_call_fn_36892inputs"Ш
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
їBЄ
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36897inputs"Ш
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
)__inference_conv2d_11_layer_call_fn_36906inputs"Ш
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
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36925inputs"Ш
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
┌B╫
0__inference_max_pooling2d_11_layer_call_fn_36930inputs"Ш
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
їBЄ
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36935inputs"Ш
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
)__inference_flatten_3_layer_call_fn_36940inputs"Ш
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_36946inputs"Ш
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
(__inference_dense_10_layer_call_fn_36955inputs"Ш
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
C__inference_dense_10_layer_call_and_return_conditional_losses_36974inputs"Ш
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
)__inference_dropout_6_layer_call_fn_36979inputs"д
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
)__inference_dropout_6_layer_call_fn_36984inputs"д
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
D__inference_dropout_6_layer_call_and_return_conditional_losses_36996inputs"д
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
D__inference_dropout_6_layer_call_and_return_conditional_losses_37001inputs"д
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
╥B╧
(__inference_dense_11_layer_call_fn_37010inputs"Ш
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
C__inference_dense_11_layer_call_and_return_conditional_losses_37029inputs"Ш
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
)__inference_dropout_7_layer_call_fn_37034inputs"д
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
)__inference_dropout_7_layer_call_fn_37039inputs"д
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
D__inference_dropout_7_layer_call_and_return_conditional_losses_37051inputs"д
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
D__inference_dropout_7_layer_call_and_return_conditional_losses_37056inputs"д
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
'__inference_dense_9_layer_call_fn_37065inputs"Ш
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
B__inference_dense_9_layer_call_and_return_conditional_losses_37076inputs"Ш
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
Є	variables
є	keras_api

Їtotal

їcount"
_tf_keras_metric
c
Ў	variables
ў	keras_api

°total

∙count
·
_fn_kwargs"
_tf_keras_metric
.:,@2Adam/m/conv2d_9/kernel
.:,@2Adam/v/conv2d_9/kernel
 :@2Adam/m/conv2d_9/bias
 :@2Adam/v/conv2d_9/bias
0:.@А2Adam/m/conv2d_10/kernel
0:.@А2Adam/v/conv2d_10/kernel
": А2Adam/m/conv2d_10/bias
": А2Adam/v/conv2d_10/bias
1:/АА2Adam/m/conv2d_11/kernel
1:/АА2Adam/v/conv2d_11/kernel
": А2Adam/m/conv2d_11/bias
": А2Adam/v/conv2d_11/bias
(:&
АА2Adam/m/dense_10/kernel
(:&
АА2Adam/v/dense_10/kernel
!:А2Adam/m/dense_10/bias
!:А2Adam/v/dense_10/bias
(:&
АА2Adam/m/dense_11/kernel
(:&
АА2Adam/v/dense_11/kernel
!:А2Adam/m/dense_11/bias
!:А2Adam/v/dense_11/bias
&:$	А2Adam/m/dense_9/kernel
&:$	А2Adam/v/dense_9/kernel
:2Adam/m/dense_9/bias
:2Adam/v/dense_9/bias
0
Ї0
ї1"
trackable_list_wrapper
.
Є	variables"
_generic_user_object
:  (2total
:  (2count
0
°0
∙1"
trackable_list_wrapper
.
Ў	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
ObM
beta:0C__inference_dense_11_layer_call_and_return_conditional_losses_37029
RbP
	BiasAdd:0C__inference_dense_11_layer_call_and_return_conditional_losses_37029
ObM
beta:0C__inference_dense_11_layer_call_and_return_conditional_losses_36566
RbP
	BiasAdd:0C__inference_dense_11_layer_call_and_return_conditional_losses_36566
ObM
beta:0C__inference_dense_10_layer_call_and_return_conditional_losses_36974
RbP
	BiasAdd:0C__inference_dense_10_layer_call_and_return_conditional_losses_36974
ObM
beta:0C__inference_dense_10_layer_call_and_return_conditional_losses_36529
RbP
	BiasAdd:0C__inference_dense_10_layer_call_and_return_conditional_losses_36529
PbN
beta:0D__inference_conv2d_11_layer_call_and_return_conditional_losses_36925
SbQ
	BiasAdd:0D__inference_conv2d_11_layer_call_and_return_conditional_losses_36925
PbN
beta:0D__inference_conv2d_11_layer_call_and_return_conditional_losses_36497
SbQ
	BiasAdd:0D__inference_conv2d_11_layer_call_and_return_conditional_losses_36497
PbN
beta:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_36887
SbQ
	BiasAdd:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_36887
PbN
beta:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_36472
SbQ
	BiasAdd:0D__inference_conv2d_10_layer_call_and_return_conditional_losses_36472
ObM
beta:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_36849
RbP
	BiasAdd:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_36849
ObM
beta:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_36447
RbP
	BiasAdd:0C__inference_conv2d_9_layer_call_and_return_conditional_losses_36447
Bb@
sequential_3/conv2d_9/beta:0 __inference__wrapped_model_36396
EbC
sequential_3/conv2d_9/BiasAdd:0 __inference__wrapped_model_36396
CbA
sequential_3/conv2d_10/beta:0 __inference__wrapped_model_36396
FbD
 sequential_3/conv2d_10/BiasAdd:0 __inference__wrapped_model_36396
CbA
sequential_3/conv2d_11/beta:0 __inference__wrapped_model_36396
FbD
 sequential_3/conv2d_11/BiasAdd:0 __inference__wrapped_model_36396
Bb@
sequential_3/dense_10/beta:0 __inference__wrapped_model_36396
EbC
sequential_3/dense_10/BiasAdd:0 __inference__wrapped_model_36396
Bb@
sequential_3/dense_11/beta:0 __inference__wrapped_model_36396
EbC
sequential_3/dense_11/BiasAdd:0 __inference__wrapped_model_36396Я
 __inference__wrapped_model_36396{+,:;OP^_mn8в5
.в+
)К&
input_4         pp
к "1к.
,
dense_9!К
dense_9         ╝
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36887t+,7в4
-в*
(К%
inputs         $$@
к "5в2
+К(
tensor_0           А
Ъ Ц
)__inference_conv2d_10_layer_call_fn_36868i+,7в4
-в*
(К%
inputs         $$@
к "*К'
unknown           А╜
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36925u:;8в5
.в+
)К&
inputs         

А
к "5в2
+К(
tensor_0         А
Ъ Ч
)__inference_conv2d_11_layer_call_fn_36906j:;8в5
.в+
)К&
inputs         

А
к "*К'
unknown         А║
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36849s7в4
-в*
(К%
inputs         pp
к "4в1
*К'
tensor_0         ll@
Ъ Ф
(__inference_conv2d_9_layer_call_fn_36830h7в4
-в*
(К%
inputs         pp
к ")К&
unknown         ll@м
C__inference_dense_10_layer_call_and_return_conditional_losses_36974eOP0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Ж
(__inference_dense_10_layer_call_fn_36955ZOP0в-
&в#
!К
inputs         А
к ""К
unknown         Ам
C__inference_dense_11_layer_call_and_return_conditional_losses_37029e^_0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Ж
(__inference_dense_11_layer_call_fn_37010Z^_0в-
&в#
!К
inputs         А
к ""К
unknown         Ак
B__inference_dense_9_layer_call_and_return_conditional_losses_37076dmn0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Д
'__inference_dense_9_layer_call_fn_37065Ymn0в-
&в#
!К
inputs         А
к "!К
unknown         н
D__inference_dropout_6_layer_call_and_return_conditional_losses_36996e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_6_layer_call_and_return_conditional_losses_37001e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_6_layer_call_fn_36979Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_6_layer_call_fn_36984Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ан
D__inference_dropout_7_layer_call_and_return_conditional_losses_37051e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_7_layer_call_and_return_conditional_losses_37056e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_7_layer_call_fn_37034Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_7_layer_call_fn_37039Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         А▒
D__inference_flatten_3_layer_call_and_return_conditional_losses_36946i8в5
.в+
)К&
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Л
)__inference_flatten_3_layer_call_fn_36940^8в5
.в+
)К&
inputs         А
к ""К
unknown         Аё
"__inference_internal_grad_fn_37190╩√№Ав}
vвs

 
)К&
result_grads_0         А
)К&
result_grads_1         А
К
result_grads_2 
к "?Ъ<

 
#К 
tensor_1         А
К
tensor_2 ё
"__inference_internal_grad_fn_37217╩¤■Ав}
vвs

 
)К&
result_grads_0         А
)К&
result_grads_1         А
К
result_grads_2 
к "?Ъ<

 
#К 
tensor_1         А
К
tensor_2 ё
"__inference_internal_grad_fn_37244╩ ААв}
vвs

 
)К&
result_grads_0         А
)К&
result_grads_1         А
К
result_grads_2 
к "?Ъ<

 
#К 
tensor_1         А
К
tensor_2 ё
"__inference_internal_grad_fn_37271╩БВАв}
vвs

 
)К&
result_grads_0         А
)К&
result_grads_1         А
К
result_grads_2 
к "?Ъ<

 
#К 
tensor_1         А
К
tensor_2 М
"__inference_internal_grad_fn_37298хГДУвП
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
"__inference_internal_grad_fn_37325хЕЖУвП
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
"__inference_internal_grad_fn_37352хЗИУвП
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
"__inference_internal_grad_fn_37379хЙКУвП
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
tensor_2 Й
"__inference_internal_grad_fn_37406тЛМСвН
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
"__inference_internal_grad_fn_37433тНОСвН
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
"__inference_internal_grad_fn_37460тПРСвН
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
tensor_2 М
"__inference_internal_grad_fn_37487хСТУвП
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
"__inference_internal_grad_fn_37514хУФУвП
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
tensor_2 ё
"__inference_internal_grad_fn_37541╩ХЦАв}
vвs

 
)К&
result_grads_0         А
)К&
result_grads_1         А
К
result_grads_2 
к "?Ъ<

 
#К 
tensor_1         А
К
tensor_2 ё
"__inference_internal_grad_fn_37568╩ЧШАв}
vвs

 
)К&
result_grads_0         А
)К&
result_grads_1         А
К
result_grads_2 
к "?Ъ<

 
#К 
tensor_1         А
К
tensor_2 ї
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_36897еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╧
0__inference_max_pooling2d_10_layer_call_fn_36892ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ї
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_36935еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╧
0__inference_max_pooling2d_11_layer_call_fn_36930ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_36859еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_9_layer_call_fn_36854ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╔
G__inference_sequential_3_layer_call_and_return_conditional_losses_36602~+,:;OP^_mn@в=
6в3
)К&
input_4         pp
p

 
к ",в)
"К
tensor_0         
Ъ ╔
G__inference_sequential_3_layer_call_and_return_conditional_losses_36652~+,:;OP^_mn@в=
6в3
)К&
input_4         pp
p 

 
к ",в)
"К
tensor_0         
Ъ г
,__inference_sequential_3_layer_call_fn_36681s+,:;OP^_mn@в=
6в3
)К&
input_4         pp
p

 
к "!К
unknown         г
,__inference_sequential_3_layer_call_fn_36710s+,:;OP^_mn@в=
6в3
)К&
input_4         pp
p 

 
к "!К
unknown         о
#__inference_signature_wrapper_36821Ж+,:;OP^_mnCв@
в 
9к6
4
input_4)К&
input_4         pp"1к.
,
dense_9!К
dense_9         