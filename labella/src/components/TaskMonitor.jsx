import React, { useEffect, useState } from "react";
import io from "socket.io-client";

const TaskMonitor = ({ resultId }) => {
  const [taskInfo, setTaskInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const socket = io(process.env.NEXT_PUBLIC_API_ENDPOINT);

    // Connect and monitor task
    socket.emit("monitor_task", { result_id: resultId });

    // Listen for updates
    socket.on("task_update", (data) => {
      setTaskInfo(data);
      console.log(data)
    });

    // Handle errors
    socket.on("error", (err) => {
      setError(err.message);
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
      {error && <p>Error: {error}</p>}
      {taskInfo ? (
        <div className="mb-8">
          <p>Job ID: {taskInfo.id}</p>
          <p>State: {taskInfo.state}</p>
          {taskInfo.state === 'PROGRESS' && (
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
          </div>)}
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
};

export default TaskMonitor;
