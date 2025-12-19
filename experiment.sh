# small matrices
# python3 -m matrix_completion.cli --m 50 --n 50 --rank 2 --num-iters 500 --print-matrices
# python3 -m matrix_completion.cli --m 50 --n 50 --rank 4 --num-iters 500 --print-matrices
# python3 -m matrix_completion.cli --num-iters 1000 --m 50 --n 50 --rank 8 --num-iters 500 --print-matrices

# medium matrices
python3 -m matrix_completion.cli --m 400 --n 400 --rank 8 --num-iters 3000 --print-matrices
python3 -m matrix_completion.cli --m 400 --n 400 --rank 16 --num-iters 3000 --print-matrices
# python3 -m matrix_completion.cli --m 400 --n 400 --rank 32 --num-iters 3000 --print-matrices

# large matrices
python3 -m matrix_completion.cli --m 800 --n 800 --rank 8 --num-iters 5000
python3 -m matrix_completion.cli --m 800 --n 800 --rank 32 --num-iters 5000
# python3 -m matrix_completion.cli --m 800 --n 800 --rank 64 --num-iters 5000