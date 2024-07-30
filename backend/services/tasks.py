from celery import shared_task 
from time import sleep
from celery.contrib.abortable import AbortableTask

@shared_task(bind=True)
def long_running_task(self, iterations) -> int:
    result = 0
    for i in range(iterations):
        result += i
        sleep(2) 
        self.update_state(state='PROGRESS',
                          meta={'current': i})
    return result 
