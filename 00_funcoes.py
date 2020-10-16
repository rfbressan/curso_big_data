# Funções recebem o endereço da variável
# Se alterar na função, a variável estará
# alterada fora dela

def altera_nota (notas):
    for i in range(0, len(notas)):
        notas[i] += 10
    return notas

notas = [10, 9, 8, 6]
print(notas)

n_notas = altera_nota(notas)
print('Notas originais: ' + str(notas))
print('Notas alteradas: ' + str(n_notas))

# O parâmetro pode ser passado por cópia da seguinte maneira
nn_notas = altera_nota(notas[:])
print('Notas originais: ' + str(notas))
print('Notas alteradas: ' + str(nn_notas))

# Número variável de argumentos como tupla
def imprime_notas (nome, *notas):
    print(nome)
    print(notas)
    for nota in notas:
        print('Nota: ' + str(nota))

imprime_notas('Rafael', 5, 8, 10)

# Funções podem ficar todas em outro arquivo .py e importar
import aula

aula.imprime_notas('Rafael', 6, 7, 8)

