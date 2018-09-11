import numpy as np
import tensorflow as tf
import random
import cv2

from threading import Thread, Lock

mutex = Lock()


clicked_state = 0

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


#second player network....


second_state_input = tf.placeholder(tf.float32, [None, 9], "second_state_input")
second_target_state_input = tf.placeholder(tf.float32, [None, 9], "second_target_state_input")
#first layer
weight_stddev = (2.0/9)**0.5
second_predict_w1 = tf.get_variable("second_predict_w1", (9, 80), initializer=tf.random_uniform_initializer())
second_predict_b1 = tf.Variable(tf.zeros(80), name="second_predict_b1")
second_predict_layer_1_output = tf.nn.leaky_relu(tf.matmul(second_state_input, second_predict_w1)+second_predict_b1)



weight_stddev = (2.0/9)**0.5
second_target_w1 = tf.get_variable("second_target_w1", (9, 80), initializer=tf.random_uniform_initializer())
second_target_b1 = tf.Variable(tf.zeros(80), name="second_target_b1")
second_target_layer_1_output = tf.nn.leaky_relu(tf.matmul(second_target_state_input, second_target_w1)+second_target_b1)


#second layer...
weight_stddev = (2.0/80)**0.5
#9 available actions...
second_predict_w2 = tf.get_variable("second_predict_w2", (80, 50), initializer=tf.random_uniform_initializer())
second_predict_b2 = tf.Variable(tf.zeros(50), name="second_predict_b2")
second_predict_layer_2_output = tf.nn.leaky_relu(tf.matmul(second_predict_layer_1_output, second_predict_w2)+second_predict_b2)



weight_stddev = (2.0/80)**0.5
second_target_w2 = tf.get_variable("second_target_w2", (80, 50), initializer=tf.random_uniform_initializer())
second_target_b2 = tf.Variable(tf.zeros(50), name="second_target_b2")
second_target_layer_2_output = tf.nn.leaky_relu(tf.matmul(second_target_layer_1_output, second_target_w2)+second_target_b2)


#third layer
weight_stddev = (2.0/50)**0.5
second_predict_w3 = tf.get_variable("second_predict_w3", (50, 9), initializer=tf.random_uniform_initializer())
second_predict_b3 = tf.Variable(tf.zeros(9), name="second_predict_b3")
second_predict_layer_3_output = tf.nn.leaky_relu(tf.matmul(second_predict_layer_2_output, second_predict_w3)+second_predict_b3)
second_max_q_action = tf.argmax(second_predict_layer_3_output, axis=1)


weight_stddev = (2.0/50)**0.5
second_target_w3 = tf.get_variable("second_target_w3", (50, 9), initializer=tf.random_uniform_initializer())
second_target_b3 = tf.Variable(tf.zeros(9), name="second_target_b3")
second_target_layer_3_output = tf.nn.leaky_relu(tf.matmul(second_target_layer_2_output, second_target_w3)+second_target_b3)




second_target_max = tf.reduce_max(second_target_layer_3_output, reduction_indices=1)


second_assign_1 = tf.assign(second_target_w1, second_predict_w1)
second_assign_2 = tf.assign(second_target_w2, second_predict_w2)
second_assign_5 = tf.assign(second_target_w3, second_predict_w3)
second_assign_3 = tf.assign(second_target_b1, second_predict_b1)
second_assign_4 = tf.assign(second_target_b2, second_predict_b2)
second_assign_6 = tf.assign(second_target_b3, second_predict_b3)



second_reward = tf.placeholder(tf.float32, [None])

second_action = tf.placeholder(tf.int32, [None])

second_one_hot = tf.one_hot(second_action, 9, 1.0, 0.0)
second_curr_q = tf.reduce_sum(second_one_hot * second_predict_layer_3_output, reduction_indices=1);

second_target_max_q = second_target_max

epsilon = 0.48
learning_rate = 0.3
second_discount_factor = 0.9


second_discounted = second_discount_factor*second_target_max_q
second_future = second_reward+second_discounted
second_difference = second_future - second_curr_q

second_kare =  tf.reduce_mean(tf.squared_difference(second_future, second_curr_q))

second_loss = second_kare

second_optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
second_optimize_nn = optimizer.minimize(second_loss)






train_saver = tf.train.Saver()
initialize = tf.global_variables_initializer()


sess = tf.Session()
sess.run(initialize)
#Networks are identical...second_
sess.run([assign_1, assign_2, assign_3, assign_4, assign_5, assign_6])
sess.run([second_assign_1, second_assign_2, second_assign_3, second_assign_4, second_assign_5, second_assign_6])




def show_game(envir):
    world = np.array([symbols[int(index)+1] for index in environment.reshape((9))], np.str).reshape(3, 3)
    print world

def sample_action():
    global environment
    if random.random() < epsilon:
        return np.random.choice(9, 1)
        pass
        pass
    else:
        return sess.run(max_q_action, feed_dict={state_input: [environment]})


def second_sample_action():
    global environment
    if random.random() < epsilon:
        return np.random.choice(9, 1)
        pass
        pass
    else:
        return sess.run(second_max_q_action, feed_dict={second_state_input: [environment]})

def check_win_state(env):
    env = env.copy().reshape((3, 3))
    user_wins = False
    program_wins = False
    for i in range(3):
        if np.sum(env[i, :])  == 3:
            user_wins = True
            program_wins = False
            #print "Row ", i, "mactch user ", env[i, :]

            break;
        if np.sum(env[:, i]) == 3:
            user_wins = True
            program_wins = False
            #print "Column ", i, "mactch user ", env[:, i]
            break;

        if np.sum(env[i, :])  == -3:
            user_wins = False
            program_wins = True
            #print "Row ", i, "mactch nn ", env[i, :]
            break;
        if np.sum(env[:, i])  == -3:
            user_wins = False
            program_wins = True
            #print "Column ", i, "mactch n ", env[:, i]
            break;

    if env[0][0] == 1 and env[1][1] == 1 and env[2][2] ==1:
        user_wins = True
        program_wins = False
        #print "Cross 0, 4, 8 mactch user"

    if env[0][2] == 1 and env[1][1] == 1 and env[2][0] == 1:
        user_wins = True
        program_wins = False
        #print "Cross 2, 4, 6 mactch user"


    if env[0][0] == -1 and env[1][1] == -1 and env[2][2] == -1:
        user_wins = False
        program_wins = True
        #print "Cross 0, 4, 8 mactch nn"

    if env[0][2] == -1 and env[1][1] == -1 and env[2][0] == -1:
        user_wins = False
        program_wins = True
        #print "Cross 2, 4, 6 mactch nn"


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
second_game_batch = []





def train_nn():

    global game_batch
    np.random.shuffle(game_batch)

    rewards = [ game_batch[i][3] for i in range(0, len(game_batch)) ]

    current_states =[ game_batch[i][0] for i in range(0, len(game_batch)) ]
    next_states = [ game_batch[i][2] for i in range(0, len(game_batch)) ]
    actions = [ game_batch[i][1][0] for i in range(0, len(game_batch)) ]

    #print "rewards : ", rewards
    #print "current_state : ", current_states
    #print "actions : ", actions
    #print "next states : ", next_states


    # "Rewards : \n", rewards
    #print "target output : \n", sess.run(target_layer_3_output, feed_dict={target_state_input: next_states })
    #print "current : \n", sess.run(predict_layer_3_output, feed_dict={state_input: current_states})
    #print "actions : \n", actions
    #print "max next q : \n", sess.run(target_max_q, feed_dict={target_state_input: next_states})
    #print "hot vector\n", sess.run(one_hot, feed_dict={action: actions})
    #out = sess.run(curr_q, feed_dict={state_input: current_states, action: actions})
    #print "qurr_q : \n", out
    #print "Out shape : ", out.shape
    #print "discounted : \n", sess.run( discounted, feed_dict={target_state_input: next_states})
    #print "future : \n", sess.run(future, feed_dict={target_state_input: next_states, reward: rewards})
    #print "difference : \n", sess.run(difference, feed_dict={state_input: current_states, target_state_input: next_states, reward: rewards, action: actions})
    #print "kare : \n", sess.run(kare, feed_dict={state_input: current_states, target_state_input: next_states, reward: rewards, action: actions})


    print "\nLoss : \n", sess.run(loss, feed_dict={state_input: current_states, target_state_input: next_states, reward: rewards, action: actions})


    for i in range(0, 20):
        sess.run(optimize_nn, feed_dict={state_input: current_states, target_state_input: next_states, reward: rewards, action: actions})

    print "\nLoss after train : ", sess.run(loss, feed_dict={state_input: current_states, target_state_input: next_states,
                                                 reward: rewards, action: actions})
    sess.run([assign_1, assign_2, assign_3, assign_4, assign_5, assign_6])


def second_train_nn():

    global second_game_batch
    np.random.shuffle(second_game_batch)

    rewards = [ second_game_batch[i][3] for i in range(0, len(second_game_batch)) ]

    current_states =[ second_game_batch[i][0] for i in range(0, len(second_game_batch)) ]
    next_states = [ second_game_batch[i][2] for i in range(0, len(second_game_batch)) ]
    actions = [ second_game_batch[i][1][0] for i in range(0, len(second_game_batch)) ]

    #print "rewards : ", rewards
    #print "current_state : ", current_states
    #print "actions : ", actions
    #print "next states : ", next_states


    # "Rewards : \n", rewards
    #print "target output : \n", sess.run(target_layer_3_output, feed_dict={target_state_input: next_states })
    #print "current : \n", sess.run(predict_layer_3_output, feed_dict={state_input: current_states})
    #print "actions : \n", actions
    #print "max next q : \n", sess.run(target_max_q, feed_dict={target_state_input: next_states})
    #print "hot vector\n", sess.run(one_hot, feed_dict={action: actions})
    #out = sess.run(curr_q, feed_dict={state_input: current_states, action: actions})
    #print "qurr_q : \n", out
    #print "Out shape : ", out.shape
    #print "discounted : \n", sess.run( discounted, feed_dict={target_state_input: next_states})
    #print "future : \n", sess.run(future, feed_dict={target_state_input: next_states, reward: rewards})
    #print "difference : \n", sess.run(difference, feed_dict={state_input: current_states, target_state_input: next_states, reward: rewards, action: actions})
    #print "kare : \n", sess.run(kare, feed_dict={state_input: current_states, target_state_input: next_states, reward: rewards, action: actions})


    print "\nsecond_Loss : \n", sess.run(second_loss, feed_dict={second_state_input: current_states, second_target_state_input: next_states, second_reward: rewards, second_action: actions})


    for i in range(0, 20):
        sess.run(second_optimize_nn, feed_dict={second_state_input: current_states, second_target_state_input: next_states, second_reward: rewards, second_action: actions})

    print "\nsecond_Loss after train : ", sess.run(second_loss, feed_dict={second_state_input: current_states, second_target_state_input: next_states,
                                                                    second_reward: rewards, second_action: actions})
    sess.run([second_assign_1, second_assign_2, second_assign_3, second_assign_4, second_assign_5, second_assign_6])




symbols = ['X', " ", 'O']
finish = False
game_over = True
nn_score, user_score = 0, 0
times = 0;

while finish == False:
    if nn_score == 10000:
        train_saver.save(sess, "nn_trained/tictoeplayer")
        print "Training has finished...."
        break


    if len(game_batch) > 100:
        train_nn()
        del game_batch
        game_batch = []

    if len(second_game_batch)>100:
        #if ( user_score - 20 > nn_score):
            second_train_nn()
            del second_game_batch
            second_game_batch = []


    if game_over == True:
        #zemin = zemin_original.copy()

        environment = np.zeros((9), dtype=np.float32)
        first_current_state = environment.copy()
        first_current_move = np.random.choice(9, 1)#sample_action()

        environment[first_current_move] = -1.0
        first_next_state = environment
        first_current_reward = 0.0
        game_over = False


    else:
        first_current_state = environment.copy()
        first_current_move =  sample_action()

        #check action result whether win...

        #If nn makes wrong move,
        if environment[first_current_move] != 0:
            #print "first nn played wrong move ", first_current_move, environment[first_current_move]
            environment[first_current_move] = -1.0
            first_next_state = environment


            game_over = True
            first_current_reward = -10
            game_batch.append([first_current_state, first_current_move, first_next_state, first_current_reward])
            continue

        environment[first_current_move] = -1.0


        first_next_state = environment




       # print "Check", check_win_state(environment)

        user_win, program_win = check_win_state(environment)
        #if program wins award it..
        if user_win == True and program_win==True:
            #print "Draw"
            #first_current_reward = 10
            #game_batch.append([first_current_state, first_current_move, first_next_state, first_current_reward])
            game_over = True

            #second_current_reward = 10
            #second_game_batch.append([second_current_state, second_current_move, first_next_state, second_current_reward])

            continue

        elif program_win == True:
            first_current_reward = 10
            nn_score += 1
            game_batch.append([first_current_state, first_current_move, first_next_state, first_current_reward])
            game_over = True

            #second_current_reward = -100
            #second_game_batch.append([second_current_state, second_current_move, first_next_state, second_current_reward])

            continue

        #second_current_reward = -0.5
        #second_game_batch.append([second_current_state, second_current_move, first_next_state, second_current_reward])

    #print "first NN played action : ", first_current_move

    #show_game(environment)
    #print "\nSecond NN turn..\n"






    #check user win...
    # if user wins, big negative reinforcement...
    second_current_state = environment.copy()
    second_current_move = second_sample_action()

    if environment[second_current_move] != 0:
        #print "second nn played wrong move ", second_current_move, environment[second_current_move]
        environment[second_current_move] = 1.0
        second_next_state = environment

        game_over = True
        second_current_reward = -10
        second_game_batch.append([second_current_state, second_current_move, second_next_state, second_current_reward])

        continue

    environment[second_current_move] = 1.0
    second_next_state = environment
    user_win, program_win = check_win_state(environment)


    #If oppenent wins, punish it...
    if user_win == True and program_win == True:
        #print "Draw"

        #second_current_reward = 10
        #second_game_batch.append([second_current_state, second_current_move, second_next_state, second_current_reward])
        game_over = True

        #first_current_reward = 10
        #game_batch.append([first_current_state, first_current_move, second_next_state, first_current_reward])

        continue

    elif user_win == True:
        user_score += 1
        #print "User has won"
        second_current_reward = 10
        second_game_batch.append([second_current_state, second_current_move, second_next_state, second_current_reward])
        game_over = True

        #first_current_reward = -100
        #game_batch.append([first_current_state, first_current_move, second_next_state, first_current_reward])



        continue

    #first_current_reward = -0.5
    #game_batch.append([first_current_state, first_current_move, second_next_state, first_current_reward])

    times += 1
    if times %10:
        pass
        print "User score : ", user_score
        print "Nn score : ", nn_score




