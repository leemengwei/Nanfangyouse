import React, { useState, useEffect, useRef } from 'react';
import EditTable, { TableContext } from '@/components/editTable';
import {
  Checkbox,
  Button,
  Form,
  Input,
  InputNumber,
  Row,
  Col,
  Table,
  Spin,
  message,
  Space,
  Upload
} from 'antd';;
import { connect } from 'dva';
import { columns3_1 } from '@/utils/data';
import {fix2} from '@/utils/fn';
import request from '@/utils/request';
import styles from '../index.less';
import selfStyle from './index.less';

function renderDiffData(value, jsx = true) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    if (jsx) {
      return null;
    }
    return [null, null];
  }
  if (value === 0) {
    if (jsx) {
      return <span>0</span>
    }
    return [0, null];
  } else if (value > 0) {
    if (jsx) {
      return <span style={{ color: '#3f8600' }}>{'+' + value}</span>
    }
    return ['+' + value, '#3f8600'];
  } else {
    if (jsx) {
      return <span style={{ color: '#cf1322' }}>{value}</span>
    }
    return [value, '#cf1322'];
  }
}
function calcDiffData(mainList, diffList) {
  return mainList.map((item) => {
    const id = item.number;
    const obj = { key: `k${item.number}` }
    const item2 = diffList.find(item => item.number === id);
    for(let i = 0; i < columns3_1.length;i++) {
      let key = columns3_1[i].dataIndex;
      if (key === 'material' || key === 'number' || key === 'name') {
        obj[key] = item[key];
      } else {
        const k2 = item2?.[key] ?? 0;
        const k1 = item?.[key] ?? 0;
        obj[key] = fix2(k1 - k2);
      }
    }
    return obj;
  })
}
const columns3_2 = [
  {
    title: '物料分类',
    dataIndex: 'material',
  },
  {
    title: '物料编码',
    dataIndex: 'number',
    //editable: true,
  },
  {
    title: '物料名称',
    dataIndex: 'name',
  },
  // {
  //   title: '上期结存干量(t)',
  //   dataIndex: 'lastBalanceDry',
  //   render: renderDiffData,
  // },
  // {
  //   title: '上期结存Cu(t)',
  //   dataIndex: 'lastBalanceCu',
  //   render: renderDiffData,
  // },
  // {
  //   title: '上期结存Ag(g)',
  //   dataIndex: 'lastBalanceAg',
  //   render: renderDiffData,
  // },
  // {
  //   title: '上期结存Au(g)',
  //   dataIndex: 'lastBalanceAu',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入干量(t)',
  //   dataIndex: 'currentIncomeDry',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入Cu(%)',
  //   dataIndex: 'currentIncomePercentageCu',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入Cu(t)',
  //   dataIndex: 'currentIncomeCu',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入Ag(g/t)',
  //   dataIndex: 'currentIncomeUnitageAg',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入Ag(g)',
  //   dataIndex: 'currentIncomeAg',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入Au(g/t)',
  //   dataIndex: 'currentIncomeUnitageAu',
  //   render: renderDiffData,
  // },
  // {
  //   title: '本期收入Au(g)',
  //   dataIndex: 'currentIncomeAu',
  //   render: renderDiffData,
  // },
  {
    title: '本期结存 干量(t)',
    dataIndex: 'currentBalanceDry',
    render: renderDiffData,
  },
  {
    title: '本期结存Cu(%)',
    dataIndex: 'currentBalancePercentageCu',
    render: renderDiffData,
  },
  {
    title: '本期结存Cu(t)',
    dataIndex: 'currentBalanceCu',
    render: renderDiffData,
  },
  {
    title: '本期结存Ag(g/t)',
    dataIndex: 'currentBalanceUnitageAg',
    render: renderDiffData,
  },
  {
    title: '本期结存Ag(g)',
    dataIndex: 'currentBalanceAg',
    render: renderDiffData,
  },
  {
    title: '本期结存Au(g/t)',
    dataIndex: 'currentBalanceUnitageAu',
    render: renderDiffData,
  },
  {
    title: '本期结存Au(g)',
    dataIndex: 'currentBalanceAu',
    render: renderDiffData,
  },
  {
    title: '本期使用 干量(t)',
    dataIndex: 'currentCostDry',
    render: renderDiffData,
  },
  {
    title: '本期使用Cu(t)',
    dataIndex: 'currentCostCu',
    render: renderDiffData,
  },
  {
    title: '本期使用Ag(g)',
    dataIndex: 'currentCostAg',
    render: renderDiffData,
  },
  {
    title: '本期使用Au(g)',
    dataIndex: 'currentCostAu',
    render: renderDiffData,
  },

]

function P(props) {
  const {  
    pinghengOldList,
    pinghengDataList,
    pinghengParameter,
    pinghengResizeDataList, 
    dispatch,
  } = props;
 
  const [diffData, setDiffData] = useState(calcDiffData(pinghengOldList, pinghengResizeDataList))
  useEffect(() => {
    setDiffData(calcDiffData(pinghengOldList, pinghengResizeDataList));
  }, [pinghengResizeDataList, pinghengOldList])  

  return (
    <div className={selfStyle.wrapper}>
      <div className={styles.row}>
        {/* <Space>
          <Button
            type="primary"
            onClick={() => {
              setShowDiff(true)
            }}
          >
            数据对比
            </Button>
        </Space> */}
      </div>

      <div className={`${styles.row} ${selfStyle.tableWrapper}`}>
        <Table
          key="number"
          columns={columns3_2}
          dataSource={diffData}
          pagination={false}
          bordered
        />
        <Row className={styles.row} style={{ marginTop: '20px' }}>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="金回收率(%)"
              value={renderDiffData(pinghengParameter.recoveryAu_ - pinghengParameter.recoveryAu, false)[0]}
            // onChange={(e) => {
            //   setInitialDataParameter('recoveryAu', e.target.value)
            // }}
            />
          </Col>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="银回收率(%)"
              value={renderDiffData(pinghengParameter.recoveryAg_ - pinghengParameter.recoveryAg, false)[0]}
            // onChange={(e) => {
            //   setInitialDataParameter('recoveryAg', e.target.value)
            // }}
            />
          </Col>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="铜回收率(%)"
              value={renderDiffData(pinghengParameter.recoveryCu_ - pinghengParameter.recoveryCu, false)[0]}
            // onChange={(e) => {
            //   setInitialDataParameter('recoveryCu', e.target.value)
            // }}
            />
          </Col>
        </Row>
      </div>
    </div>
  )
}

function mapStateToProps(state) {
  const { 
    config,  
    pinghengOldList,
    pinghengDataList,
    pinghengParameter,
    pinghengResizeDataList, 
  } = state.global;
  return {
    config,
    pinghengOldList,
    pinghengDataList,
    pinghengParameter,
    pinghengResizeDataList,
  };
}
export default connect(mapStateToProps)(P);