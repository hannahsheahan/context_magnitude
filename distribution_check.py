import numpy as np
import random
import matplotlib.pyplot as plt

N = 10000
fullrange, lowrange, highrange = [[] for i in range(3)]
for i in range(N):
    fullrange.append(random.randint(1,15))
    lowrange.append(random.randint(1,10))
    highrange.append(random.randint(6,15))

# difference code
fulldiff, lowdiff, highdiff = [[] for i in range(3)]
allranges = [fullrange, lowrange, highrange]
abrecord = np.zeros((3,N,2))

for r in range(len(allranges)):
    for i in range(N):
        a = random.choice(allranges[r])
        b = random.choice(allranges[r])
        while a == b:
            b = random.choice(allranges[r])
        if r == 0:
            fulldiff.append(a-b)
        elif r==1:
            lowdiff.append(a-b)
        elif r==2:
            highdiff.append(a-b)
        abrecord[r,i,0] = a
        abrecord[r,i,1] = b



plt.figure()
fig, ax = plt.subplots(1,3)
ax[0].hist(fullrange, bins=15, color='gold')
ax[0].set_title('full range')
ax[1].hist(lowrange, bins=10, color='orangered')
ax[1].set_title('low range')
ax[2].hist(highrange, bins=10, color='dodgerblue')
ax[2].set_title('high range')
ax[0].set_xlim(0,15)
ax[1].set_xlim(0,15)
ax[2].set_xlim(0,15)

plt.figure()
fig, ax = plt.subplots(1,3)
ax[0].hist(fulldiff, color='gold', bins=29)
ax[1].hist(lowdiff, color='orangered', bins=19)
ax[2].hist(highdiff, color='dodgerblue', bins=19)

ax[0].set_xlim(-15,15)
ax[1].set_xlim(-15,15)
ax[2].set_xlim(-15,15)
ax[0].set_title('full range')
ax[0].set_xlabel('A - B')
ax[1].set_title('low range')
ax[1].set_xlabel('A - B')
ax[2].set_title('high range')
ax[2].set_xlabel('A - B')
ax[0].set_ylim(0,1100)
ax[1].set_ylim(0,1100)
ax[2].set_ylim(0,1100)


plt.figure()
plt.hist(fulldiff, color='gold', bins=29, alpha=0.3)
plt.hist(lowdiff, color='orangered', bins=19, alpha=0.3)
plt.hist(highdiff, color='dodgerblue', bins=19, alpha=0.3)
plt.xlim(-15,15)


# And finally, if we need to represent ALL these differences, the distribution over different differences becomes...
# Remember that we only have one decoder.

plt.figure()
allnumbers = [fulldiff[:], lowdiff[:], highdiff[:]]
allnumbers = [a for sublist in allnumbers for a in sublist]
plt.hist(allnumbers, color='green', bins=29)
plt.title('all 3 number ranges combined')
plt.xlabel('A - B')


# Also, what would even one of these difference codes look like if we averaged over all the number Bs?
#meanAcount = np.zeros(np.max(fullrange))
#for i in range(np.max(fullrange)):  # for each number A
#    for j in range(len(fullrange)):
#        if
#    meanAcount[i]



# Now if we were going to allocate neural resource efficiently to representing that entire distribution, how would we do it?
