import React, { useState } from 'react';
import Arrow from './Arrow';
const ImageSlider = ({ images, bucketname, bucketprefix, handleImageChange }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const handlePrev = () => {
    const newIndex = currentIndex === 0 ? images.length - 1 : currentIndex - 1;
    setCurrentIndex(newIndex)
    handleImageChange(newIndex)
};

  const handleNext = () => {
    const newIndex = currentIndex === images.length - 1 ? 0 : currentIndex + 1;
    setCurrentIndex(newIndex);
    handleImageChange(newIndex);
};


  return (
    <div>
      <div className='flex gap-4 items-center justify-center'>
        <button onClick={handlePrev}><Arrow direction="left"/></button>
        <img src={`https://${bucketname}.s3.amazonaws.com/${bucketprefix}/${images[currentIndex].data}`} alt={`Slide ${currentIndex}`} className='w-96 h-64 rounded-lg'/>
        <button onClick={handleNext}><Arrow direction="right"/></button>
      </div>
    </div>
  );
};

export default ImageSlider;
