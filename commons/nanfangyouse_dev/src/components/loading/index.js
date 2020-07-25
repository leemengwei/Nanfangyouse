import React from 'react';
import { Spin } from 'antd';


export default function () {
  return (
    <div>
      <div style={{ paddingTop: 100, textAlign: 'center' }}>
        <Spin size="large" />
      </div>
    </div>
  )
}