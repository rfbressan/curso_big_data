# Sequencia de Fibonacci

fibo = [0, 1]

def fib(termos):
    if termos == 1:
        print([0])
    elif termos == 2:
        print(fibo)
    else:
        fibo.append(fibo[-2]+fibo[-1])
        fib(termos - 1)

n = int(input('Número de termos: '))
fib(n)