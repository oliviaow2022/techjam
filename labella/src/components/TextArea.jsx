'use client';
import React, {useState} from 'react';

const TextArea = ({label, name, value, onChange, error}) => {

    return(
        <div>
            <label htmlFor="input-box" className='block text-white'>{label}</label>
            <textarea 
                id={name}
                name={name}
                type="type" 
                value={value}
                onChange={onChange} 
                className={`text-white p-2 border border-white border-opacity-50 rounded-lg h-32 w-64 bg-transparent`}
            />
            {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
    )
};

export default TextArea;