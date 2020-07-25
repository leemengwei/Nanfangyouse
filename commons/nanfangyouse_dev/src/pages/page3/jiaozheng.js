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
} from 'antd';

import { connect } from 'dva';
import { columns3_1 } from '@/utils/data';
import {objNumerFix} from '@/utils/fn';
import request from '@/utils/request';
import styles from '../index.less';
import selfStyle from './index.less';

function P(props) {
  const { 
    config, 
    materialOptions,
    pinghengSetList, 
    pinghengDataList, 
    pinghengParameter, 
    pinghengResizeDataList,
    dispatch 
  } = props;
  
  function setResizeDataParameter(value) {
    dispatch({
      type: 'global/changePinghengParameter',
      value
    })
  }
  function setResizeData_(list) {
    dispatch({
      type: 'global/changePinghengResizeDataList',
      list
    })
  }

  // 数据矫正
  function correctData() {
    const payload = {
      list: pinghengDataList,
      setting: pinghengSetList,
      parameter: {
        recoveryAu: pinghengParameter.recoveryAu,
        recoveryAg: pinghengParameter.recoveryAg,
        recoveryCu: pinghengParameter.recoveryCu
      }
    }
    console.log(payload)
    request({
      method: 'POST',
      host: config.host,
      port: config.port,
      url: 'correct_data',
      payload,
      cb: (res) => {
        const o = {
          recoveryAu_: res.parameter.recoveryAu,
          recoveryAg_: res.parameter.recoveryAg,
          recoveryCu_: res.parameter.recoveryCu,
        };
        setResizeData_(res.list.map(item => objNumerFix(item)));
        setResizeDataParameter(o);
      }
    })
  }

  return (
    <div className={selfStyle.wrapper}>
      <div className={styles.row}>
        <Space>

          <Button
            type="primary"
            onClick={correctData}
          >
            数据矫正
            </Button>
        </Space>
      </div>
      <div
        className={`${styles.row} ${selfStyle.tableWrapper}`}>
        <TableContext.Provider value={{
          columns: columns3_1,
          dataSource: pinghengResizeDataList,
          materialOptions: materialOptions,
          setData: setResizeData_
        }}>
          <EditTable />
        </TableContext.Provider >
        <Row className={styles.row} style={{ marginTop: '20px' }}>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="金回收率(%)"
              value={pinghengParameter.recoveryAu_}
              onChange={(e) => {
                setResizeDataParameter({'recoveryAu_': e.target.value})
              }}
            />
          </Col>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="银回收率(%)"
              value={pinghengParameter.recoveryAg_}
              onChange={(e) => {
                setResizeDataParameter({'recoveryAg_': e.target.value})
              }}
            />
          </Col>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="铜回收率(%)"
              value={pinghengParameter.recoveryCu_}
              onChange={(e) => {
                setResizeDataParameter({'recoveryCu_': e.target.value})
              }}
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
    materialOptions,
    pinghengDataList, 
    pinghengSetList, 
    pinghengParameter,
    pinghengResizeDataList,
  } = state.global;
  return {
    config,
    materialOptions,
    pinghengSetList,
    pinghengDataList,
    pinghengParameter,
    pinghengResizeDataList,
  };
}
export default connect(mapStateToProps)(P);