import React, { useState, useEffect } from 'react';
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
import XLSX from 'xlsx';
//import { UploadOutlined } from '@ant-design/icons';
import { connect } from 'dva';
//import {workbook2blob, openDownloadDialog} from '@/utils/fn';
import { columns3_1 } from '@/utils/data';
import request from '@/utils/request';
import {handleSettingData, fix2, objNumerFix} from '@/utils/fn';
import styles from '../index.less';
import selfStyle from './index.less';

function P(props) {
  const { config, pinghengParameter, materialOptions, dispatch } = props;
  const [initialData, setInitialData] = useState(null);

  useEffect(() => {
    dispatch({
      type: 'global/changeConfig',
      port: 7002
    })
  }, [])
  function setMaterialOptions(list) {
    dispatch({
      type: 'global/changeMaterialOptions',
      list
    })
  }

  function setInitialDataParameter(value) {
    dispatch({
      type: 'global/changePinghengParameter',
      value
    })
  }
  function setInitialData_(data, first) {
    setInitialData(data);
    if (first) {
      const arr = handleSettingData(data);
      dispatch({
        type: 'global/changePinghengSetList',
        dataList: data,
        settingList: arr
      })
    }
    dispatch({
      type: 'global/changePinghengOldList',
      list: data
    })
  }

  // 导入excel
  function customRequest(e) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const data = e.target.result;
      const xlsxData = XLSX.read(data, { type: 'binary' });
      const value = XLSX.utils.sheet_to_json(xlsxData.Sheets[xlsxData.SheetNames[0]]);
      const optionsList = Array(value.length);
      const value_ = value.map((item, index) => {
        optionsList[index] = item.material;
        return objNumerFix(item);
      })
      console.log(value_)
      setInitialData_(value_, true);
      setInitialDataParameter({
        recoveryAu: value_[0].recoveryAu ?? 0,
        recoveryAg: value_[0].recoveryAg ?? 0,
        recoveryCu: value_[0].recoveryCu ?? 0
      })
      setMaterialOptions(Array.from(new Set(optionsList)));
    }
    reader.readAsBinaryString(e.file);
  }
  return (
    <div className={selfStyle.wrapper}>
      <div className={styles.row}>
        <Space>
          <Upload
            showUploadList={false}
            customRequest={customRequest}
          >
            <Button
              type="primary"
            >
              数据录入
            </Button>
          </Upload>
          <Input style={{ width: '250px' }} addonBefore="当前请求地址" value={config.host} onChange={(e) => {
            dispatch({
              type: 'global/changeConfig',
              host: e.target.value
            })
          }} />
          <Input style={{ width: '150px' }} addonBefore="当前端口" value={config.port} onChange={(e) => {
            dispatch({
              type: 'global/changeConfig',
              port: e.target.value
            })
          }} />
        </Space>
      </div>
      <div
        className={`${styles.row} ${selfStyle.tableWrapper}`}>
        <TableContext.Provider value={{
          columns: columns3_1,
          dataSource: initialData,
          materialOptions: materialOptions,
          setData: setInitialData_
        }}>
          <EditTable />
        </TableContext.Provider >
        <Row className={styles.row} style={{ marginTop: '20px' }}>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="金回收率(%)"
              value={pinghengParameter.recoveryAu}
              onChange={(e) => {
                setInitialDataParameter({ 'recoveryAu': Number(e.target.value) })
              }}
            />
          </Col>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="银回收率(%)"
              value={pinghengParameter.recoveryAg}
              onChange={(e) => {
                setInitialDataParameter({ 'recoveryAg': Number(e.target.value) })
              }}
            />
          </Col>
          <Col span={6}>
            <Input
              style={{ width: '250px' }}
              addonBefore="铜回收率(%)"
              value={pinghengParameter.recoveryCu}
              onChange={(e) => {
                setInitialDataParameter({ 'recoveryCu': Number(e.target.value) })
              }}
            />
          </Col>
        </Row>
      </div>
    </div>
  )
}

function mapStateToProps(state) {
  const { config, pinghengParameter, materialOptions } = state.global;
  return {
    config,
    pinghengParameter,
    materialOptions,
  };
}
export default connect(mapStateToProps)(P);