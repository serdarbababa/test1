# Bouncing Ball Simulator
# By @TokyoEdTech
# Part 1: Getting Started
import turtle
from turtle import Turtle, Screen

def simpleBOuncingBal():
    wn = turtle.Screen()
    wn.bgcolor("black")
    wn.title("Bouncing Ball Simulator")


    ball = turtle.Turtle()
    ball.shape("circle")
    ball.color("green")
    ball.penup()
    ball.speed(0)
    ball.goto(0, 200)
    ball.dy = 0

    gravity = 0.1

    while True:

        ball.dy -= gravity
        ball.sety(ball.ycor() + ball.dy)


        # Check for a bounce
        if ball.ycor() < -300:
            ball.dy *= -1
def controlArrows():


    screen = Screen()

    screen.setup(500, 500)
    screen.title("Turtle Keys")

    move = Turtle(shape="turtle")

    def k1():
        move.forward(10)

    def k2():
        move.left(45)

    def k3():
        move.right(45)

    def k4():
        move.backward(10)

    screen.onkey(k1, "Up")
    screen.onkey(k2, "Left")
    screen.onkey(k3, "Right")
    screen.onkey(k4, "Down")

    screen.listen()

    screen.exitonclick()





from modules.Components import Abstract, Context1, Context2, Actuator, Spektron



ops = generateOperations(500, Test=False, verbose=True)
# print((ops))

decodedOps = decodeOperationsDeeper(input_data=ops, len_data=len(ops), qsignals=quantize(symbols, len_of_data=4))

# GG,counter = train_tree( mergeList(mergeList(decodedOps)), GG, counter , WL, overlap)
# print(decodedOps)

ids = getBranchId(mergeList(mergeList(decodedOps)), GG, counter, 8, overlap)

print(ids)
ops = mergeList(ops)

#controlArrows()