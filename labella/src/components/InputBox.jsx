'use client';
import React, {useState} from 'react';

const InputBox = ({label, name, value, onChange, error}) => {

    return(
        <div>
            <label htmlFor="input-box" className='block'>{label}</label>
            <input 
                id={name}
                name={name}
                type="text" 
                value={value}
                onChange={onChange} 
                className='text-white p-2 border rounded-lg h-8 bg-transparent'    
            />
            {error && <p className="text-red-500">{error}</p>}
        </div>
    )
};

export default InputBox;