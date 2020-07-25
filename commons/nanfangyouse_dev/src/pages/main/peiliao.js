import React from 'react';
import { Tabs, Divider, Card } from 'antd';
import Page1 from '@/pages/page1';
import Page2 from '@/pages/page2';
// import Page3 from '@/pages/page3';
// import styles from './index.css';

const { TabPane } = Tabs;
export default function() {

  return (
    <div>
      <Divider style={{ marginTop: '5px' }} />
      <h1 style={{ textAlign: 'center', fontWeight: '900' }}>南方有色配料项目</h1>
      <Tabs
        defaultActiveKey="1"
        size="large"
        tabBarStyle={{ paddingLeft: '50px' }}
        style={{ padding: '0 20px' }}
      >
        <TabPane tab="新增配方" key="1">
          <Card>
            <Page1
            />
          </Card>
        </TabPane>
        <TabPane tab="配方衔接" key="2">
          <Card>
            <Page2 />
          </Card>
        </TabPane>
        {/* <TabPane tab="金属平衡" key="3">
          <Card>
            <Page3 />
          </Card>
        </TabPane> */}
      </Tabs>
    </div>
  )
}
