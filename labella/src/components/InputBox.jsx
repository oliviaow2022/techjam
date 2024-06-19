'use client';
import React, {useState} from 'react';

const InputBox = ({label}) => {
    const [inputValue, setInputValue] = useState('');
    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    };

    const handleSubmit = () => {
    };

    return(
        <div>
            <label htmlFor="input-box" className='block'>{label}</label>
            <input type="text" value={inputValue} onChange={handleInputChange} className='text-white p-2 border rounded-lg h-8 bg-transparent'></input>
        </div>
    )
};

export default InputBox;