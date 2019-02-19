import matplotlib.pyplot as plt
import pandas as pd


#To plot the charts from csv output

df = pd.read_csv("Easy Policy.csv")
df1 = pd.read_csv("Easy Value.csv")
qdf = pd.read_csv("Easy Q-Learning L0.1 q100.0 E0.5.csv")

df.set_index("iter")
df1.set_index('iter')
qdf.set_index('iter')

#time
plt.plot(df.index,df['time'])
plt.plot(df.index,df1['time'])
plt.xlabel('Iterations')
plt.ylabel('Time (s)')
plt.title("PI VI Cumulative Time vs Iterations")
plt.legend(['Policy Iteration', 'Value Iteration'] )
plt.show()

plt.clf()

#Reward
plt.plot(df.index,df['reward'])
plt.plot(df.index,df1['reward'])
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title("PI VI Reward vs Iterations")
plt.legend(['Policy Iteration', 'Value Iteration'] )
plt.show()
plt.clf()

#Steps
plt.plot(df.index,df['steps'])
plt.plot(df.index,df1['steps'])
plt.xlabel('Iterations')
plt.ylabel('Steps')
plt.title("PI VI Steps vs Iterations")
plt.legend(['Policy Iteration', 'Value Iteration'] )
plt.show()
plt.clf()


#Converge


plt.plot(df.index,df['convergence'])
plt.plot(df.index,df1['convergence'])
plt.xlabel('Iterations')
plt.ylabel('Convergence')
plt.title("PI VI Convergence vs Iterations")
plt.legend(['Policy Iteration', 'Value Iteration'] )
plt.show()
plt.clf()
#
# # #time
qdf = qdf.ix[:100,:]
plt.plot(qdf.index,qdf['time'])
plt.plot(df.index,df['time'])
plt.plot(df.index,df1['time'])
plt.xlabel('Iterations')
plt.ylabel('Time (s)')
plt.title("QLearner, PI, VI  Cumulative Time vs Iterations")
plt.legend(['QLearner','Policy Iteration', 'Value Iteration'] )
plt.show()

plt.clf()

#qdf = qdf.ix[2000 :,:]

#Converge
plt.plot(qdf.index,qdf['convergence'])
plt.xlabel('Iterations')
plt.ylabel('Convergence')
plt.title("QLearner Convergence vs Iterations")
plt.legend(['QLearner'] )
plt.show()
plt.clf()

#reward
plt.plot(qdf.index,qdf['reward'])
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title("QLearner Reward vs Iterations")
plt.legend(['QLearner'] )
plt.show()
plt.clf()