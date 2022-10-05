export JULIA_PROJECT=$1
echo julia project is: $JULIA_PROJECT
shift
julia $JULIA_PROJECT/scripts/main.jl $@
