import React, { useEffect, useState } from "react";
import io from "socket.io-client";

const TaskMonitor = ({ resultId, onSuccess }) => {
  const [taskInfo, setTaskInfo] = useState(null);

  useEffect(() => {
    const socket = io(process.env.NEXT_PUBLIC_API_ENDPOINT);

    // Connect and monitor task
    socket.emit("monitor_task", { result_id: resultId });
    console.log('connected to socket with', resultId)

    // Listen for updates
    socket.on("task_update", (data) => {
      setTaskInfo(data);
      if (data.state === "SUCCESS") {
        onSuccess();
      }
    });

    // Handle errors
    socket.on("error", (err) => {
      console.log(err.message);
    });

    // Clean up on component unmount
    return () => {
      socket.disconnect();
    };
  }, [resultId]);

  // Calculate progress percentage
  const calculateProgress = (epoch, num_epochs) => {
    if (num_epochs === 0) return 0;
    const progress = (epoch / num_epochs) * 100;
    return progress.toFixed(1); // Format to one decimal place
  };

  return (
    <div>
      {taskInfo && (
        <div className="mb-8">
          {taskInfo && taskInfo.id ? (
            <p>Job ID: {taskInfo.id}</p>
          ) : (
            <p>Job ID: {resultId}</p>
          )}
          <p>State: {taskInfo.state}</p>
          {taskInfo.state === "PROGRESS" && (
            <div className="w-full bg-gray-200 rounded-full h-4 mt-2">
              <div
                style={{
                  width: `${calculateProgress(
                    taskInfo.info.epoch,
                    taskInfo.info.num_epochs
                  )}%`,
                }}
                className="bg-green-500 h-full rounded-full text-center text-white text-xs leading-4"
              >
                {`${calculateProgress(
                  taskInfo.info.epoch,
                  taskInfo.info.num_epochs
                )}%`}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TaskMonitor;
