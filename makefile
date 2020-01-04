#
# minimal makefile
#
# compilation rules
CC = gcc
CFLAGS = -Wall -Ofast -fopenmp
#CFLAGS = -Wall -Ofast 
LDFLAGS = -lm -lgsl

# source, object, and include (.h) files
SRCS = main.c image.c model_disk.c evolve.c
OBJS = main.o image.o model_disk.o evolve.o
#SRCS = main.c image.c model_uniform.c evolve.c
#OBJS = main.o image.o model_uniform.o evolve.o
INCS = noisy.h

noisy: $(OBJS) $(INCS) makefile
	$(CC) $(CFLAGS) -o noisy $(OBJS) $(LDFLAGS)

$(OBJS):  $(INCS) makefile

clean:
	rm *.o
