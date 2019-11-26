from modules.Components import Abstract, Context1, Context2, Actuator, Spektron

# initialize Spektron
spek = Spektron()
spek.displaySpektron()

# train with 1000 operations
print("train with 1000 operations")
for i in range(1000):
    spek.oneBeat(verbose=False)

print("one operation")
operations = spek.getInstantOperationInput()
[print(i, operations[i]) for i in range(len(operations))]
#
print()
print("ONE RUN")
print()
for count, item in enumerate(operations):
    spek.oneBeat(symbol=item, verbose=True)

spek.displaySpektron(False)

print()
print("COMPLEX BEAT")
print()
for symbol in operations:
    spek.oneComplexBeat(symbol, verbose=True)

print()
print("CHECK OPERATION")
print()
for symbol in operations[:-1]:
    spek.checkOperation(symbol, verbose=True)


print()
print("More CHECK OPERATION")
print()

# for i in range(100):
#     operations= spek.getInstantOperationInput()
#     for symbol in operations:
#         spek.oneComplexBeat(symbol)
