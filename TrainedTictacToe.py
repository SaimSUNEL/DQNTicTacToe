import numpy as np
import tensorflow as tf
import random
import cv2

from threading import Thread, Lock

mutex = Lock()


clicked_state = 0

def mouse_callback(*args):
    global clicked_state, mutex, environment

    if args[0] == 4:
        x = 0
        y = 0
        if 10<args[1]<160:
            x=0
        elif 170<args[1]<320:
            x=1
        elif 330<args[1]<480:
            x=2

        if 10 < args[2] < 160:
            y = 0
        elif 170 < args[2] < 320:
            y = 1
        elif 330 < args[2] < 480:
            y = 2

        clicked_state = 3 * y + x

        if environment[clicked_state] != 0:
            print "Invalid selection"
            return

        display_image(3*y+x, o_image)

        cv2.imshow("Game", zemin)
        print "Shown..."

        mutex.release()






def display_image(state_count, image):
    start_x = locations[str(state_count+1)][0]
    start_y = locations[str(state_count+1)][1]
    zemin[start_y:start_y+120, start_x:start_x+120] = image[:]


locations = {"1": (25, 31), "2": (185, 31), "3": (345, 31),
             "4": (25, 185), "5": (185, 185), "6": (345, 185),
             "7": (25, 345), "8": (185, 345), "9": (345, 345)}



x_image = None
o_image = None
zemin_original = None
zemin = None


nn_score, user_score = 0, 0

x_image = cv2.imread("x.jpg")
o_image = cv2.imread("o.jpg")
zemin = cv2.imread("zemin.jpg")
zemin_original = zemin.copy()
cv2.imshow("Game", zemin)
cv2.waitKey(100)

cv2.setMouseCallback("Game", mouse_callback)


# 1. 2. 3.
# 4. 5. 6.
# 7. 8. 9.


#3x3 tic tac toe environment:
#the each squre could take 3 different values: -1 for X, 0 nothing, 1 for O
environment = np.zeros((9), dtype=np.int8)

#if the game has not ended yet, the reward will be 0
#if the game has ended, reward will be +10
#a wrong move results in high lose, -50


state_input = tf.placeholder(tf.float32, [None, 9], "state_input")
target_state_input = tf.placeholder(tf.float32, [None, 9], "target_state_input")
#first layer
weight_stddev = (2.0/9)**0.5
predict_w1 = tf.get_variable("predict_w1", (9, 80), initializer=tf.random_uniform_initializer())
predict_b1 = tf.Variable(tf.zeros(80), name="predict_b1")
predict_layer_1_output = tf.nn.leaky_relu(tf.matmul(state_input, predict_w1)+predict_b1)



weight_stddev = (2.0/9)**0.5
target_w1 = tf.get_variable("target_w1", (9, 80), initializer=tf.random_uniform_initializer())
target_b1 = tf.Variable(tf.zeros(80), name="target_b1")
target_layer_1_output = tf.nn.leaky_relu(tf.matmul(target_state_input, target_w1)+target_b1)


#second layer...
weight_stddev = (2.0/80)**0.5
#9 available actions...
predict_w2 = tf.get_variable("predict_w2", (80, 50), initializer=tf.random_uniform_initializer())
predict_b2 = tf.Variable(tf.zeros(50), name="predict_b2")
predict_layer_2_output = tf.nn.leaky_relu(tf.matmul(predict_layer_1_output, predict_w2)+predict_b2)



weight_stddev = (2.0/80)**0.5
target_w2 = tf.get_variable("target_w2", (80, 50), initializer=tf.random_uniform_initializer())
target_b2 = tf.Variable(tf.zeros(50), name="target_b2")
target_layer_2_output = tf.nn.leaky_relu(tf.matmul(target_layer_1_output, target_w2)+target_b2)


#third layer
weight_stddev = (2.0/50)**0.5
predict_w3 = tf.get_variable("predict_w3", (50, 9), initializer=tf.random_uniform_initializer())
predict_b3 = tf.Variable(tf.zeros(9), name="predict_b3")
predict_layer_3_output = tf.nn.leaky_relu(tf.matmul(predict_layer_2_output, predict_w3)+predict_b3)
max_q_action = tf.argmax(predict_layer_3_output, axis=1)


weight_stddev = (2.0/50)**0.5
target_w3 = tf.get_variable("target_w3", (50, 9), initializer=tf.random_uniform_initializer())
target_b3 = tf.Variable(tf.zeros(9), name="target_b3")
target_layer_3_output = tf.nn.leaky_relu(tf.matmul(target_layer_2_output, target_w3)+target_b3)




target_max = tf.reduce_max(target_layer_3_output, reduction_indices=1)


assign_1 = tf.assign(target_w1, predict_w1)
assign_2 = tf.assign(target_w2, predict_w2)
assign_5 = tf.assign(target_w3, predict_w3)
assign_3 = tf.assign(target_b1, predict_b1)
assign_4 = tf.assign(target_b2, predict_b2)
assign_6 = tf.assign(target_b3, predict_b3)



reward = tf.placeholder(tf.float32, [None])

action = tf.placeholder(tf.int32, [None])

one_hot = tf.one_hot(action, 9, 1.0, 0.0)
curr_q = tf.reduce_sum(one_hot * predict_layer_3_output, reduction_indices=1);

target_max_q = target_max

epsilon = 0.48
learning_rate = 0.3
discount_factor = 0.9


discounted = discount_factor*target_max_q
future = reward+discounted
difference = future - curr_q

kare =  tf.reduce_mean(tf.squared_difference(future, curr_q))

loss = kare

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
optimize_nn = optimizer.minimize(loss)

train_saver = tf.train.Saver()



sess = tf.Session()

train_saver.restore(sess, "nn_trained/tictoeplayer")



test_world = np.array([-1, -1, 0, 0,1,1, 1,-1,0], dtype=np.float32)

print sess.run(predict_layer_3_output, feed_dict={state_input: [test_world]})





def show_game(envir):
    world = np.array([symbols[int(index)+1] for index in environment.reshape((9))], np.str).reshape(3, 3)
    print world

def sample_action():
    global environment
    #no random move...
    return sess.run(max_q_action, feed_dict={state_input: [environment]})

def check_win_state(env):
    env = env.copy().reshape((3, 3))
    user_wins = False
    program_wins = False
    for i in range(3):
        if np.sum(env[i, :])  == 3:
            user_wins = True
            program_wins = False
            print "Row ", i, "mactch user ", env[i, :]

            break;
        if np.sum(env[:, i]) == 3:
            user_wins = True
            program_wins = False
            print "Column ", i, "mactch user ", env[:, i]
            break;

        if np.sum(env[i, :])  == -3:
            user_wins = False
            program_wins = True
            print "Row ", i, "mactch nn ", env[i, :]
            break;
        if np.sum(env[:, i])  == -3:
            user_wins = False
            program_wins = True
            print "Column ", i, "mactch n ", env[:, i]
            break;

    if env[0][0] == 1 and env[1][1] == 1 and env[2][2] ==1:
        user_wins = True
        program_wins = False
        print "Cross 0, 4, 8 mactch user"

    if env[0][2] == 1 and env[1][1] == 1 and env[2][0] == 1:
        user_wins = True
        program_wins = False
        print "Cross 2, 4, 6 mactch user"


    if env[0][0] == -1 and env[1][1] == -1 and env[2][2] == -1:
        user_wins = False
        program_wins = True
        print "Cross 0, 4, 8 mactch nn"

    if env[0][2] == -1 and env[1][1] == -1 and env[2][0] == -1:
        user_wins = False
        program_wins = True
        print "Cross 2, 4, 6 mactch nn"


    zer = False
    for j in environment:
        if j ==0:
            zer = True


    #draw...
    if zer == True:
        pass
    else:
        if user_wins == False and program_wins == False:
            user_wins, program_wins = True, True

    return user_wins, program_wins


game_batch= []


symbols = ['X', " ", 'O']
finish = False
game_over = True
while finish == False:



    if game_over == True:
        zemin = zemin_original.copy()

        environment = np.zeros((9), dtype=np.float32)
        current_state = environment.copy()
        current_move = np.random.choice(9, 1)#sample_action()

        #program plays X
        display_image(current_move[0], x_image)
        cv2.imshow("Game", zemin)

        cv2.waitKey(100)

        environment[current_move] = -1
        next_state = environment
        current_reward = 0.0
        game_over = False
        if mutex.locked():
            mutex.release()

    else:
        current_state = environment.copy()
        current_move =  sample_action()

        #check action result whether win...

        #If nn makes wrong move,
        if environment[current_move] != 0:
            print "nn played wrong move ", current_move, environment[current_move]
            environment[current_move] = -1
            next_state = environment


            game_over = True
            current_reward = -800
            #game_batch.append([current_state, current_move, next_state, current_reward])


            continue

        environment[current_move] = -1


        next_state = environment
        display_image(current_move[0], x_image)
        cv2.imshow("Game", zemin)
        cv2.waitKey(100)



        print "Check", check_win_state(environment)

        user_win, program_win = check_win_state(environment)
        #if program wins award it..
        if user_win == True and program_win==True:
            print "Draw"
            current_reward = 20
            #game_batch.append([current_state, current_move, next_state, current_reward])
            game_over = True

            continue





        elif program_win == True:
            current_reward = 200
            nn_score += 1
            #game_batch.append([current_state, current_move, next_state, current_reward])
            game_over = True

            continue


    print "NN played action : ", current_move

    show_game(environment)
    print "\nyour turn\n"

    while mutex.locked() == True:
        cv2.waitKey(250)


    mutex.acquire()
    while mutex.locked() == True:


        cv2.waitKey(250)

    #check user win...
    # if user wins, big negative reinforcement...
    user_move = clicked_state

    environment[user_move] = 1
    next_state = environment
    user_win, program_win = check_win_state(environment)


    #If oppenent wins, punish it...
    if user_win == True and program_win == True:
        print "Draw"
        current_reward = 20
        #game_batch.append([current_state, current_move, next_state, current_reward])
        game_over = True

        continue

    elif user_win == True:
        user_score += 1
        print "User has won"
        current_reward = -200
        #game_batch.append([current_state, current_move, next_state, current_reward])
        game_over = True

        continue

    current_reward = -40
    #game_batch.append([current_state, current_move, next_state, current_reward])

    print "User score : ", user_score
    print "Nn score : ", nn_score




