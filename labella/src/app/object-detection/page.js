'use client'
import { useState, useEffect, useRef } from 'react'

export default function ObjectDetection() {{
    const canvasRef = useRef(null);
    const image_path = 'https://miro.medium.com/v2/resize:fit:1400/1*v0Bm-HQxWtpbQ0Yq463uqw.jpeg'
    const [rectangles, setRectangles] = useState([]);
    const [isDrawing, setIsDrawing] = useState(false);
    const [currentRect, setCurrentRect] = useState(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        const image = new Image();

        image.src = image_path
        image.onload = () => {
            context.drawImage(image, 0, 0, canvas.width, canvas.height);
            rectangles.forEach(({ x, y, width, height, color }) => {
                drawRectangle(context, x, y, width, height, color);
            });

            if (currentRect) {
                drawRectangle(context, currentRect.x, currentRect.y, currentRect.width, currentRect.height);
            }
        }
    }, [rectangles, currentRect])

    const drawRectangle = (context, x, y, width, height, color) => {
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.strokeRect(x, y, width, height);
    };

    const handleMouseDown = (event) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        setIsDrawing(true);
        setCurrentRect({ x, y, width: 0, height: 0 })
    }

    const handleMouseMove = (event) => {
        if (!isDrawing) return;
    
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
    
        const newRect = {
          x: currentRect.x,
          y: currentRect.y,
          width: x - currentRect.x,
          height: y - currentRect.y,
        };
    
        setCurrentRect(newRect);
    };

    const handleMouseUp = () => {
        setIsDrawing(false);
        setCurrentRect(null);
        setRectangles([...rectangles, currentRect]);
    }

    return (
        <main>
            <canvas 
                ref={canvasRef}
                width={1000}
                height={500} 
                className='border border-black' 
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
            />
        </main>
    )
}}