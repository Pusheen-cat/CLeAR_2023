import copy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
import pickle as pkl
import datetime

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np

class inspect_data:
    def run(self, input, label, feature, stack, action , current_idx, in_task_idx, with_stack, length, tasks):
        dim = input.shape[2]
        print(f'Now inspecting.. After task {current_idx} of {tasks}')
        input = np.array(input) #B, length, in_dim = 3  @!! End token in the end
        label = np.array(label)  # B, out_dim
        feature = np.array(feature)  # Batch, length+1, out_feature_dim = 256
        print('input.shape', input.shape)
        print('label.shape', label.shape)
        print('feature.shape',feature.shape)
        if with_stack:
            stack = np.array(stack)
            action = np.array(action)
            print('stack.shape', stack.shape)
            print('action.shape', action.shape)
        '''
        input.shape (512, 81, 4)
        label.shape (512, 82, 3)
        feature.shape (512, 163, 256)
        stack.shape (512, 163, 1, 30, 8)
        action.shape (512, 163, 1, 3)
        input_alphabet.shape (512, 81)
        '''

        '''
        dim4: (stripe, x_1, x_2, x_3)
        x_1 = x % 2 
        x = x // 2
        x_2 = x % 2
        x = x // 2
        x_3 = x % 2
        '''
        if dim == 3:
            input_alphabet = input[:,:,0]+input[:,:,1]*2+input[:,:,2]*4
            stripe = np.zeros(input_alphabet.shape)
        if dim == 4:
            input_alphabet = input[:,:,1]+input[:,:,2]*2+input[:,:,3]*4
            stripe = input[:,:,0]



        #print(input_alphabet[0])

        '''
        feature: Batch, length, 
        '''


        #print('input_alphabet.shape', input_alphabet.shape)
        feat = feature[:,1:,:].reshape(-1, feature.shape[-1])
        #print(feat.shape) #(41472, 256)
        #print('ori_input_alphabet',ori_input_alphabet.shape) #(512, 81)
        #oia = ori_input_alphabet.reshape(-1)
        #print('ori_input_alphabet', oia.shape) #(41472,)

        if current_idx == 0:  # or current_idx == 1:
            self.pca = decomposition.PCA(n_components=3)
            self.pca.fit(feat)
        X = self.pca.transform(feat)


        #print(X.shape) #(41472, 3)
        #print(oia)

        if not with_stack:
            if in_task_idx == 1: #this is EP : Whether same start-end
                ori_input_alphabet = np.where(input_alphabet > 3, 1, 0)  # B, length
                adj_oia = []
                for batch in range(ori_input_alphabet.shape[0]):
                    adj_oia_onebatch = []
                    for idx_ in range(len(ori_input_alphabet[batch])):
                        if idx_ == 0:
                            adj_oia_onebatch.append(ori_input_alphabet[batch][idx_].item())
                        else:
                            if ori_input_alphabet[batch][idx_].item() == 1:
                                if ori_input_alphabet[batch][idx_-1].item() ==1:
                                    adj_oia_onebatch.append(1)
                                else:
                                    adj_oia_onebatch.append(2)
                            else:
                                if ori_input_alphabet[batch][idx_-1].item() ==1:
                                    adj_oia_onebatch.append(3)
                                else:
                                    adj_oia_onebatch.append(0)
                    adj_oia.append(adj_oia_onebatch)
                adj_oia = np.array(adj_oia)
                print('adj_oia.shape',adj_oia.shape)
                adj_oia = adj_oia.reshape(-1)
                markers = ["*", "x", "o", "s"]
                label_ = ['0-0', '1-1', '0-1', '1-0']
                for i, marker in enumerate(markers):
                    x_val = X[adj_oia == i][:,0]
                    y_val = X[adj_oia == i][:,1]
                    plt.scatter(x_val, y_val, marker=marker, s= 1, label=label_[i])
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
                plt.close()

                adj_oia = []
                for batch in range(ori_input_alphabet.shape[0]):
                    adj_oia_onebatch = []
                    for idx_ in range(len(ori_input_alphabet[batch])):
                        if idx_ == 0:
                            adj_oia_onebatch.append(ori_input_alphabet[batch][idx_].item())
                            init  = ori_input_alphabet[batch][idx_].item()
                        else:
                            if ori_input_alphabet[batch][idx_].item() == 1:
                                if init == 1:
                                    adj_oia_onebatch.append(1)
                                else:
                                    adj_oia_onebatch.append(2)
                            else:
                                if init == 1:
                                    adj_oia_onebatch.append(3)
                                else:
                                    adj_oia_onebatch.append(0)
                    adj_oia.append(adj_oia_onebatch)
                adj_oia = np.array(adj_oia)
                print('adj_oia.shape', adj_oia.shape)
                adj_oia = adj_oia.reshape(-1)
                markers = ["*", "x", "o", "s"]
                label_ = ['0-0', '1-1', '0-1', '1-0']
                for i, marker in enumerate(markers):
                    x_val = X[adj_oia == i][:, 0]
                    y_val = X[adj_oia == i][:, 1]
                    plt.scatter(x_val, y_val, marker=marker, s=1, label=label_[i])
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
                plt.close()

            if in_task_idx == 1: #This is for PC : Sum of seq even/odd
                ori_input_alphabet = np.where(input_alphabet > 3, 1, 0)  # B, length
                adj_oia = []
                for batch in range(ori_input_alphabet.shape[0]):
                    adj_oia_onebatch = []
                    for idx_ in range(len(ori_input_alphabet[batch])):
                        label_ = ori_input_alphabet[batch][:idx_+1]
                        adj_oia_onebatch.append(np.sum(label_).item()%2)
                    adj_oia.append(adj_oia_onebatch)
                adj_oia = np.array(adj_oia)
                adj_oia = adj_oia.reshape(-1)
                markers = ["*", "x"]
                label_ = ['even', 'odd']
                for i, marker in enumerate(markers):
                    x_val = X[adj_oia == i][:, 0]
                    y_val = X[adj_oia == i][:, 1]
                    plt.scatter(x_val, y_val, marker=marker, s=1, label=label_[i])
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
                plt.close()
                #'''
                adj_oia = []
                for batch in range(ori_input_alphabet.shape[0]):
                    adj_oia_onebatch = []
                    for idx_ in range(len(ori_input_alphabet[batch])):
                        label_ = ori_input_alphabet[batch][:idx_]
                        if np.sum(label_)% 2 == 0:
                            if ori_input_alphabet[batch][idx_] ==0:
                                adj_oia_onebatch.append(0)
                            else:
                                adj_oia_onebatch.append(1)
                        else:
                            if ori_input_alphabet[batch][idx_] ==0:
                                adj_oia_onebatch.append(2)
                            else:
                                adj_oia_onebatch.append(3)
                    adj_oia.append(adj_oia_onebatch)
                adj_oia = np.array(adj_oia)
                adj_oia = adj_oia.reshape(-1)
                markers = ["*", "x", "*", "x"]
                label_ = ['even+0', 'even+1', 'odd+0', 'odd+1',]
                for i, marker in enumerate(markers):
                    x_val = X[adj_oia == i][:, 0]
                    y_val = X[adj_oia == i][:, 1]
                    plt.scatter(x_val, y_val, marker=marker, s=1, label=label_[i])
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
                plt.close()
                #'''


            if in_task_idx == 3: #This is for CN :
                ori_input_alphabet = np.where(input_alphabet > 5, 1, np.where(input_alphabet > 2, 0, -1))  # B, length
                adj_oia = []
                for batch in range(ori_input_alphabet.shape[0]):
                    adj_oia_onebatch = []
                    for idx_ in range(len(ori_input_alphabet[batch])):
                        label_ = ori_input_alphabet[batch][:idx_+1]
                        adj_oia_onebatch.append(np.sum(label_).item()%5)
                    adj_oia.append(adj_oia_onebatch)
                adj_oia = np.array(adj_oia)
                adj_oia = adj_oia.reshape(-1)
                markers = ["*", "x", "*", "x", "*"]
                label_ = ['0', '1', '2', '3', '4',]
                for i, marker in enumerate(markers):
                    x_val = X[adj_oia == i][:, 0]
                    y_val = X[adj_oia == i][:, 1]
                    plt.scatter(x_val, y_val, marker=marker, s=1, label=label_[i])
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
                plt.close()


        if with_stack:
            if tasks[in_task_idx] == 'stack_manipulation':
                #@ Stack Manipulation --> must in_task_idx with real Stack Manipulation sample order
                '''
                input.shape (512, 81, 4)
                label.shape (512, 82, 3)
                feature.shape (512, 163, 256) --> feat = feature[:,1:,:].reshape(-1, feature.shape[-1])
                stack.shape (512, 163, 1, 30, 8)
                action.shape (512, 163, 1, 3) #0:push / 1:pop / 2:non-op
                input_alphabet.shape (512, 81)
                '''
                now = datetime.datetime.now()
                with open(f"/home/bong/code/research/formal_language/neural_networks_chomsky_hierarchy/inspection/{now.hour}_{now.minute}_{current_idx}_{in_task_idx}_len{length}_data.pkl", "wb") as fw:
                    pkl.dump(input, fw)
                    pkl.dump(label, fw)
                    pkl.dump(feature, fw)
                    pkl.dump(stack, fw)
                    pkl.dump(action, fw)
                    pkl.dump(input_alphabet, fw)


                idx_action = np.argmax(stripe, axis=1) # stripe
                input_adj = []
                for batch_idx in range(input_alphabet.shape[0]):
                    input_b_f = np.where(input_alphabet[batch_idx][:idx_action[batch_idx]]>3, 1, 0)
                    input_b_b = np.where(input_alphabet[batch_idx][idx_action[batch_idx]:]>4, 4,np.where(input_alphabet[batch_idx][idx_action[batch_idx]:]>1,3,2))
                    input_b = np.concatenate((input_b_f,input_b_b), axis=None)
                    input_adj.append(input_b)
                    #print(action[batch_idx, idx_action[batch_idx], 0, 0])
                input_adj = np.array(input_adj) #B, length (item: 0,1,2,3,4)
                #print(input_adj.shape) #(512, 81)
                in_len = input_adj.shape[1]

                print(np.sum(input_adj == 0)) #add0
                print(np.sum(input_adj == 1)) #add1
                print(np.sum(input_adj == 2)) #pop
                print(np.sum(input_adj == 3)) #push 0
                print(np.sum(input_adj == 4)) #push 1
                # #0:push / 1:pop / 2:non-op


                init_value = 1
                total_prob = []
                for n_stack in range(1):
                    prob = []
                    for idx_ in range(6):
                        sub_prob = []
                        for idx__ in range(3):
                            if idx_ < 5:
                                sub_prob.append(np.sum(
                                    action[:, init_value:init_value + in_len, n_stack, idx__] * [input_adj == idx_]) / np.sum(
                                    input_adj == idx_))
                            else:
                                sub_prob.append(np.mean(action[:, init_value + in_len:, 0, idx__]))
                        prob.append(sub_prob)
                    prob = np.array(prob)
                    total_prob.append(prob)
                total_prob = np.array(total_prob)
                print('init_value 1',total_prob)

            if tasks[in_task_idx] == 'stack_manipulation': # @ Bucket Sort
                '''
                                input.shape (512, 81, 4)
                                label.shape (512, 82, 3)
                                feature.shape (512, 163, 256) --> feat = feature[:,1:,:].reshape(-1, feature.shape[-1])
                                stack.shape (512, 163, 1, 30, 8)
                                action.shape (512, 163, 1, 3) #0:push / 1:pop / 2:non-op
                                input_alphabet.shape (512, 81)
                                '''

                input_adj = np.where(input_alphabet >3, np.where(input_alphabet >5, 3, 2),  np.where(input_alphabet >1, 1, 0))
                # print(input_adj.shape) #(512, 81)
                in_len = input_adj.shape[1]

                print(np.sum(input_adj == 0))
                print(np.sum(input_adj == 1))
                print(np.sum(input_adj == 2))
                print(np.sum(input_adj == 3))

                init_value = 1
                total_prob = []
                for n_stack in range(1):
                    prob = []
                    for idx_ in range(5):
                        sub_prob = []
                        for idx__ in range(3):
                            if idx_ < 4:
                                sub_prob.append(np.sum(
                                    action[:, init_value:init_value + in_len, n_stack, idx__] * [input_adj == idx_]) / np.sum(
                                    input_adj == idx_))
                            else:
                                sub_prob.append(np.mean(action[:, init_value + in_len:, 0, idx__]))
                        prob.append(sub_prob)
                    prob = np.array(prob)
                    total_prob.append(prob)
                total_prob = np.array(total_prob)
                print('init_value 1', total_prob)
                action  # B,length,3