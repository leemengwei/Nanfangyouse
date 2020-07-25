import React from 'react';
import { Tabs, Divider, Card } from 'antd';
import Luru from '@/pages/page3/luru';
import Shezhi from '@/pages/page3/shezhi';
import Jiaozheng from '@/pages/page3/jiaozheng';
import Duibi from '@/pages/page3/duibi';
import Daochu from '@/pages/page3/daochu';
// import styles from './index.css';

const { TabPane } = Tabs;
export default function() {

  return (
    <div>
      <Divider style={{ marginTop: '5px' }} />
      <h1 style={{ textAlign: 'center', fontWeight: '900' }}>南方金属平衡项目</h1>
      <Tabs
        defaultActiveKey="1"
        size="large"
        tabBarStyle={{ paddingLeft: '50px' }}
        style={{ padding: '0 20px' }}
      >
        <TabPane tab="数据录入" key="1">
          <Card>
            <Luru />
          </Card>
        </TabPane>
        <TabPane tab="数据设置" key="2">
          <Card>
            <Shezhi />
          </Card>
        </TabPane>
        <TabPane tab="数据矫正" key="3">
          <Card>
            <Jiaozheng />
          </Card>
        </TabPane>
        <TabPane tab="数据对比" key="4">
          <Card>
            <Duibi />
          </Card>
        </TabPane>
        <TabPane tab="数据导出" key="5">
          <Card>
            <Daochu />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}
