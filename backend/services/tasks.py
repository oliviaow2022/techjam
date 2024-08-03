from celery import shared_task 
from time import sleep
from celery.contrib.abortable import AbortableTask

@shared_task(bind=True, base=AbortableTask)
def long_running_task(self, iterations) -> int:
    result = 0
    for i in range(iterations):
        result += i
        self.update_state(state='PROGRESS',
                          meta={'current': i})
        sleep(2) 

    return result 
