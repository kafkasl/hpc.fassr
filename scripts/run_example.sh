export PYTHONPATH=$PYTHONPATH:$(pwd)/../src

export ComputingUnits=2
runcompss $1 \
    --python_interpreter=python3 \
    -dtm \
    $(pwd)/../src/fassr.py \
        --debug \
        -k=15 \
        --datasets=normal \
        --trade_mode=sell_all \
        --trade_frequency=52 \
        --train_period=53
