#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX 10240
#define NUM_THREADS  10

int n1,n2;
char *s1,*s2;
FILE *fp;
int countArray[NUM_THREADS]={0};

/* read input file and generate string s1/s2 and length n1/n2 */
int readf(FILE *fp)
{
    if((fp=fopen("strings.txt", "r"))==NULL){
    	printf("ERROR: can't open string.txt!\n");
    	return 0;
    }
    s1=(char *)malloc(sizeof(char)*MAX);
    if(s1==NULL){
    	printf("ERROR: Out of memory!\n");
    	return -1;
    }
    s2=(char *)malloc(sizeof(char)*MAX);
    if(s1==NULL){
    	printf("ERROR: Out of memory\n");
    	return -1;
    }
    /* read s1 s2 from the file */
    s1=fgets(s1, MAX, fp);
    s2=fgets(s2, MAX, fp);
    n1=strlen(s1);  /* length of s1 */
    n2=strlen(s2)-1; /* length of s2 */
    
    if(s1==NULL || s2==NULL || n1<n2)  /* when error exit */
    	return -1;
    return 0;
}

int num_substring(int t)
{
    /* add your logic here */
    
    int i, part_size = (n1 / NUM_THREADS)+1, 
	t_begin_seek = t * part_size, 
	t_end_seek = t_begin_seek + part_size;

    /* 1, how to distribute different parts of string s1 into different threads */
    
    for (i = t_begin_seek; i < t_end_seek && ((n1-i)-n2) >= 0; i++) {
	int j, mtch = 0;
	for (j = 0; j < n2; j++) {
	    if(*(s1+i+j) == *(s2+j)) {
		countArray[t] += (++mtch == n2); 
	    } else {  
		break; 
	    }
        }
    }

    /* 2, how to sum up the total number of substring from all threads */

    return countArray[t];
}

void *calSubStringThread(void *threadid){
    long tid = (long)threadid;

    printf("This is thread %ld, ", tid);
    int num = num_substring(tid);
    printf("find num of is: %d\n", num);

    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    pthread_t threads[NUM_THREADS];
    int t, rc;
    int totalNum = 0;

    readf(fp);

    for(t=0; t<NUM_THREADS; t++){
        rc = pthread_create(&threads[t], NULL, calSubStringThread, (void *) (size_t)t);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for(t=0; t<NUM_THREADS; t++){
        pthread_join(threads[t], NULL);
	totalNum+=countArray[t];
    }

    printf("The number of substrings is: %d\n", totalNum);
    return 1;
}
