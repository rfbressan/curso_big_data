def imprime_notas (nome, *notas):
    print('Nome do aluno: ' + nome)
    print(notas)
    for i in range(0, len(notas)):
        print('Nota ' + str(i+1) + ' : ' + str(notas[i]))