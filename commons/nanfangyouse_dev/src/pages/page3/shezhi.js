import React, { useState, useRef, useEffect} from 'react';
import EditTable, { TableContext } from '@/components/editTable';
import {
  Checkbox,
  Button,
  Form,
  Input,
  Row,
  Col,
  Table,
  Space,

} from 'antd';

import { connect } from 'dva';
import request from '@/utils/request';
import {handleSettingData} from '@/utils/fn';
import styles from '../index.less';
import selfStyle from './index.less';



function P(props) {
  const { config, pinghengOldList,pinghengSetList, dispatch } = props;
  
  const [data, setData] = useState(pinghengSetList);

  let prevCountRef = useRef([...data]);
  useEffect(() => {
    prevCountRef.current = [...data];
  }, [data])
  useEffect(() => {
    setData(handleSettingData(pinghengOldList))
  }, [pinghengOldList])
  function setData_(data){
    setData(data);
    const deleteNumbers = [];
    const settingList = data.filter(item => {
      const bol = !!item.delete;
      if (bol) {
        deleteNumbers.push(item.number); 
      }
      return !bol;
    })
    const dataList = pinghengOldList.filter(item => !deleteNumbers.includes(item.number))
    dispatch({
      type: 'global/changePinghengSetList',
      settingList,
      dataList
    })
  }

  const columns = [
    {
      title: '排除',
      dataIndex: 'delete',
      render: (text, record, index) => <Checkbox
        checked={text}
        onChange={() => {
          const newData = [...prevCountRef.current]
          newData[index] = { ...record }
          newData[index].delete = !text
          setData_(newData)
        }}
      />
    },
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
    {
      title: '本期结存干量 最大值',
      dataIndex: 'currentBalanceDryMax',
      editable: true,
    },
    {
      title: '本期结存干量 最小值',
      dataIndex: 'currentBalanceDryMin',
      editable: true,
    },
    {
      title: '本期结存干量 方差',
      dataIndex: 'currentBalanceDryVariance',
      editable: true,
    },
    {
      title: '本期结存Cu 最大值',
      dataIndex: 'currentBalancePercentageCuMax',
      editable: true,
    },
    {
      title: '本期结存Cu 最小值',
      dataIndex: 'currentBalancePercentageCuMin',
      editable: true,
    },
    {
      title: '本期结存Cu 方差',
      dataIndex: 'currentBalancePercentageCuVariance',
      editable: true,
    },
    {
      title: '本期结存Ag 最大值',
      dataIndex: 'currentBalancePercentageAgMax',
      editable: true,
    },
    {
      title: '本期结存Ag 最小值',
      dataIndex: 'currentBalancePercentageAgMin',
      editable: true,
    },
    {
      title: '本期结存Ag 方差',
      dataIndex: 'currentBalancePercentageAgVariance',
      editable: true,
    },
    {
      title: '本期结存Au 最大值',
      dataIndex: 'currentBalancePercentageAuMax',
      editable: true,
    },
    {
      title: '本期结存Au 最小值',
      dataIndex: 'currentBalancePercentageAuMin',
      editable: true,
    },
    {
      title: '本期结存Au 方差',
      dataIndex: 'currentBalancePercentageAuVariance',
      editable: true,
    },
  ]

 

  return (
    <div className={selfStyle.wrapper}>
      {/* <div className={styles.row}>
        <Space>
         
          <Button
            type="primary"
            onClick={correctData}
          >
            数据矫正
            </Button>
        </Space>
      </div> */}
      <div
        className={`${styles.row} ${selfStyle.tableWrapper}`}>
        <TableContext.Provider value={{
          columns,
          dataSource: data,
          setData: setData_
        }}>
          <EditTable />
        </TableContext.Provider >
      </div>
    </div>
  )
}

function mapStateToProps(state) {
  const { config, pinghengOldList, pinghengSetList} = state.global;
  return {
    config,
    pinghengOldList,
    pinghengSetList,
  };
}
export default connect(mapStateToProps)(P);