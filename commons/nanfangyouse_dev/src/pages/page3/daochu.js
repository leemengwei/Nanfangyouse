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
import XLSX from 'xlsx';
//import { UploadOutlined } from '@ant-design/icons';
import { connect } from 'dva';
//import {workbook2blob, openDownloadDialog} from '@/utils/fn';
import { columns3_1 } from '@/utils/data';
import request from '@/utils/request';
import styles from '../index.less';
import selfStyle from './index.less';

function renderDiffData(value, jsx = true) {
  if (value === undefined || value === null || Number.isNaN(value)){
    if (jsx){
      return null;
    }
    return [null, null];
  }
  if (value === 0) {
    if (jsx){
      return <span>0</span>
    }
    return [0, null];
  } else if (value > 0) {
    if (jsx){
      return <span style={{ color: '#3f8600' }}>{'+' + value}</span>
    }
    return ['+' + value, '#3f8600'];
  } else {
    if (jsx){
      return <span style={{ color: '#cf1322' }}>{value}</span>
    }
    return [value, '#cf1322'];
  }
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
  {
    title: '上期结存干量(t)',
    dataIndex: 'lastBalanceDry',
    render: renderDiffData,
  },
  {
    title: '上期结存Cu(t)',
    dataIndex: 'lastBalanceCu',
    render: renderDiffData,
  },
  {
    title: '上期结存Ag(kg)',
    dataIndex: 'lastBalanceAg',
    render: renderDiffData,
  },
  {
    title: '上期结存Au(kg)',
    dataIndex: 'lastBalanceAu',
    render: renderDiffData,
  },
  {
    title: '本期收入干量(t)',
    dataIndex: 'currentIncomeDry',
    render: renderDiffData,
  },
  {
    title: '本期收入Cu(%)',
    dataIndex: 'currentIncomePercentageCu',
    render: renderDiffData,
  },
  {
    title: '本期收入Cu(t)',
    dataIndex: 'currentIncomeCu',
    render: renderDiffData,
  },
  {
    title: '本期收入Ag(g/t)',
    dataIndex: 'currentIncomeUnitageAg',
    render: renderDiffData,
  },
  {
    title: '本期收入Ag(kg)',
    dataIndex: 'currentIncomeAg',
    render: renderDiffData,
  },
  {
    title: '本期收入Au(g/t)',
    dataIndex: 'currentIncomeUnitageAu',
    render: renderDiffData,
  },
  {
    title: '本期收入Au(kg)',
    dataIndex: 'currentIncomeAu',
    render: renderDiffData,
  },
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
    title: '本期结存Ag(kg)',
    dataIndex: 'currentBalanceAg',
    render: renderDiffData,
  },
  {
    title: '本期结存Au(g/t)',
    dataIndex: 'currentBalanceUnitageAu',
    render: renderDiffData,
  },
  {
    title: '本期结存Au(kg)',
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
    title: '本期使用Ag(kg)',
    dataIndex: 'currentCostAg',
    render: renderDiffData,
  },
  {
    title: '本期使用Au(kg)',
    dataIndex: 'currentCostAu',
    render: renderDiffData,
  },

]

function P(props) {
  const { pinghengResizeDataList, pinghengParameter } = props;

  // 导出excel
  function outputExcel() {
    const result = [...pinghengResizeDataList];
    result[0].recoveryAu = pinghengParameter.recoveryAu_;
    result[0].recoveryAg = pinghengParameter.recoveryAg_;
    result[0].recoveryCu = pinghengParameter.recoveryCu_;
    const sheet1 = XLSX.utils.json_to_sheet(result);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, sheet1, "导出");
    const wopts = {
      bookType: "xlsx",
      bookSST: false,
      type: "binary"
    };
    XLSX.writeFile(wb, '导出.xlsx', wopts);
  }

  return (
    <div className={selfStyle.wrapper}>
      <div className={styles.row}>
        <Space>
          <Button
            type="primary"
            onClick={outputExcel}
          >
            数据导出
            </Button>
        </Space>
      </div>
      
    </div>
  )
}

function mapStateToProps(state) {
  const { config, pinghengResizeDataList,pinghengParameter } = state.global;
  return {
    config,
    pinghengResizeDataList,
    pinghengParameter
  };
}
export default connect(mapStateToProps)(P);