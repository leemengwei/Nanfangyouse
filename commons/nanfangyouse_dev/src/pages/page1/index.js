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
  Space,
  Table,
  Spin,
  Divider,
  Tabs
} from 'antd';
import { connect } from 'dva';
import request from '@/utils/request';
import { list1_1, columns1_1, columns1_2, columns1_3 } from '@/utils/data';
import styles from './index.less';

const { TabPane } = Tabs

const fkdata = [...list1_1];
const elementsMixtureListColumns = [...columns1_1];
const resultListColumns = [...columns1_2];
const resultElementsMixtureListColumns = [...columns1_3];

function H(props) {
  const { config, dispatch } = props;

  function onFinish(values) {
    console.log(values)
    const elementRuls = /checkbox/g

    const list = data.filter((item) => !item.delete)

    const elementsTargetList = Object.keys(values)
      .filter((item) => elementRuls.test(item) && values[item])
      .map((item) => {
        const name = item.replace('checkbox', '')
        return {
          name,
          percentage: values[name],
          priority: values[`priority${name}`]
        }
      })
    const payload = {
      list,
      presetParameter: {
        matteTargetGradePercentage: values.matteTargetGradePercentage,
        maxType: values.maxType,
        oxygenConcentration: values.oxygenConcentration,
        consumedAmount: values.consumedAmount,
        peaCoal: values.peaCoal,
        oxygenPeaCoalRatio: values.oxygenPeaCoalRatio,
        FeSiO2Ratio: values.FeSiO2Ratio,
        matteFePercentage: values.matteFePercentage,
        matteSPercentage: values.matteSPercentage,
        slagCuPercentage: values.slagCuPercentage,
        slagSPercentage: values.slagSPercentage,
        slagFePercentage: values.slagFePercentage,
        slagSiO2Percentage: values.slagSiO2Percentage,
      },
      elementsTargetList,
      modelWeight: {
        gaPop: values.gaPop,
        gaEpoch: values.gaEpoch,
        modelFactorBeta: values.modelFactorBeta,
        modelFactorAlpha: values.modelFactorAlpha,
        modelFactorGamma: values.modelFactorGamma,
      }
    }
    console.log(payload)
    setResult(null)
    setResultShow(true)
    request({
      method: 'POST',
      host: config.host,
      port: config.port,
      url: 'calculate',
      payload,
      cb: (res) => {
        setResult(res)
      }
    })
  }
  function quickUpdate() {
    request({
      method: 'POST',
      host: config.host,
      port: config.port,
      url: 'quick_update',
      payload: result,
      cb: (res) => {
        setResult(res)
      }
    })
  }
  function onFinishFailed(errorInfo) {
    console.log('Failed:', errorInfo);
  }
  function getInventory() {
    setTableLoading(true)
    request({
      method: 'GET',
      host: config.host,
      port: config.port,
      url: 'getInventory',
      cb: (res) => {
        const data = res.list.map((item, index) => ({
          ...item,
          index,
          delete: false,
          inventoryBalance: ''
        }))
        setData(data);
        setMaterialList(res.materialList);
        setTableLoading(false)
      }
    })

  }

  const [data, setData] = useState(fkdata);
  const [materialList, setMaterialList] = useState([]);
  const [result, setResult] = useState(null);
  const [resultShow, setResultShow] = useState(false);
  const [tableLoading, setTableLoading] = useState(false);
  const setResultList = (data) => {
    //console.log(data)
    setResult({ ...result, list: [...data] })
  }
  const setData_ = (data) => {
    setData(data);
    sessionStorage.setItem('page1list', JSON.stringify(data));
  }
  let prevCountRef = useRef([...data]);
  useEffect(() => {
    prevCountRef.current = [...data];
  }, [data])

  const fkcolumns = [
    {
      title: '必选',
      dataIndex: 'required',
      render: (text, record, index) => <Checkbox
        checked={text}
        onChange={() => {
          const newData = [...prevCountRef.current]
          newData[index] = { ...record }
          newData[index].required = !text
          !text && (newData[index].delete = false)
          setData_(newData)
        }}
      />
    },
    {
      title: '排除',
      dataIndex: 'delete',
      render: (text, record, index) => <Checkbox
        checked={text}
        onChange={() => {
          const newData = [...prevCountRef.current]
          newData[index] = { ...record }
          newData[index].delete = !text
          if (!text) {
            newData[index].required = false
            newData[index].clean = false
          }
          setData_(newData)
        }}
      />
    },
    {
      title: '清库',
      dataIndex: 'clean',
      render: (text, record, index) => <Checkbox
        checked={text}
        onChange={() => {
          const newData = [...prevCountRef.current]
          newData[index] = { ...record }
          newData[index].clean = !text
          if (!text) {
            newData[index].required = true
            newData[index].delete = false
          }
          setData_(newData)
        }}
      />
    },
    {
      title: '名称',
      dataIndex: 'name',
    },
    {
      title: '批次号',
      dataIndex: 'number',
    },
    {
      title: 'Cu',
      dataIndex: 'Cu',
      // editable: true,
    },
    {
      title: 'Fe',
      dataIndex: 'Fe',
      // editable: true,
    },
    {
      title: 'S',
      dataIndex: 'S',
      // editable: true,
    },
    {
      title: 'SiO2',
      dataIndex: 'SiO2',
      // editable: true,
    }, {
      title: 'CaO',
      dataIndex: 'CaO',
      // editable: true,
    },
    {
      title: 'As',
      dataIndex: 'As',
      // editable: true,
    }, {
      title: 'Zn',
      dataIndex: 'Zn',
      // editable: true,
    },
    {
      title: 'Pb',
      dataIndex: 'Pb',
      // editable: true,
    },
    {
      title: 'MgO',
      dataIndex: 'MgO',
      // editable: true,
    },
    {
      title: 'Al2O3',
      dataIndex: 'Al2O3',
      // editable: true,
    },
    {
      title: 'H2O',
      dataIndex: 'H2O',
      // editable: true,
    },
    {
      title: 'Sb',
      dataIndex: 'Sb',
    },
    {
      title: 'Bi',
      dataIndex: 'Bi',
    },
    {
      title: 'Ni',
      dataIndex: 'Ni',
    },
    {
      title: 'Ag(t/g)',
      dataIndex: 'Ag',
    },
    {
      title: 'Au(t/g)',
      dataIndex: 'Au',
    },
    {
      title: '库存/吨',
      dataIndex: 'inventory',
      // editable: true,
    },
    {
      title: '演算比例',
      dataIndex: 'calculatePercentage',
      editable: true,
      step: 0.01
    },
    // {
    //   title: '库存余量',
    //   dataIndex: 'inventoryBalance',
    // },
  ];

  const [columns] = useState(fkcolumns);

  return (
    <div>
      <div className={styles.row}>
        <Space>
          <Button type="primary" onClick={() => {
            getInventory()
          }}>获取库存</Button>
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
      <Spin spinning={tableLoading}>
        <TableContext.Provider value={{
          columns,
          dataSource: data,
          setData: setData_
        }}>
          <EditTable />
        </TableContext.Provider >
        <Table
          className={styles.block}
          rowKey={'name'}
          columns={elementsMixtureListColumns}
          dataSource={(() => {
            let value = { name: '元素加权平均含量' }
            materialList.forEach((item) => {
              value[item.name] = item.percentage
            })
            return [value]
          })()}
          pagination={false}
          bordered
        />
      </Spin>
      <div>
        <Form
          // layout="inline"
          className={styles.block}
          labelCol={{ span: 15 }}
          onFinish={onFinish}
          onFinishFailed={onFinishFailed}
        >
          <div>
            <Tabs defaultActiveKey="1" type="card">
              <TabPane tab="预设参数" key="1" forceRender>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label="冰铜目标品味(%)"
                      name="matteTargetGradePercentage"
                      initialValue={74}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="冰铜铁含量(%)"
                      name="matteFePercentage"
                      initialValue={3.5}
                      step={.01}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="冰铜硫含量(%)"
                      name="matteSPercentage"
                      initialValue={20.84}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label="渣中铜含量(%)"
                      name="slagCuPercentage"
                      initialValue={1.99}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="渣中硫含量(%)"
                      name="slagSPercentage"
                      initialValue={.45}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="渣中铁含量(%)"
                      name="slagFePercentage"
                      initialValue={48}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      //labelCol={12}
                      label="渣中二氧化硅含量(%)"
                      name="slagSiO2Percentage"
                      initialValue={24}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label={<span>粒煤单位耗氧量(Nm<sup>3</sup>/t)</span>}
                      name="oxygenPeaCoalRatio"
                      initialValue={1100}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="粒煤(t/h)"
                      name="peaCoal"
                      initialValue={1.5}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="Fe/SiO2"
                      name="FeSiO2Ratio"
                      initialValue={2}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="给矿量(t/h)"
                      name="consumedAmount"
                      initialValue={150}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label="最大类别数"
                      name="maxType"
                      initialValue={4}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="氧浓度(%)"
                      name="oxygenConcentration"
                      initialValue={85}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} />
                    </Form.Item>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="元素含量" key="2" forceRender>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxCu"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>Cu(%)</span>
                      <Form.Item
                        name="Cu"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityCu"
                        noStyle={true}
                      >
                        <InputNumber
                          placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxSiO2"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>SiO2(%)</span>
                      <Form.Item
                        name="SiO2"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="prioritySiO2"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxZn"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>Zn(%)</span>
                      <Form.Item
                        name="Zn"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityZn"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxAl2O3"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>Al2O3(%)</span>
                      <Form.Item
                        name="Al2O3"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityAl2O3"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxFe"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>Fe(%)</span>
                      <Form.Item
                        name="Fe"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityFe"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxCaO"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>CaO(%)</span>
                      <Form.Item
                        name="CaO"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityCaO"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxPb"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>Pb(%)</span>
                      <Form.Item
                        name="Pb"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityPb"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxH2O"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>H2O(%)</span>
                      <Form.Item
                        name="H2O"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityH2O"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxS"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>S(%)</span>
                      <Form.Item
                        name="S"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityS"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxAs"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>As(%)</span>
                      <Form.Item
                        name="As"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityAs"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Form.Item
                        name="checkboxMgO"
                        valuePropName="checked"
                        noStyle={true}
                      >
                        <Checkbox />
                      </Form.Item>
                      <span className={styles.elementspan}>MgO(%)</span>
                      <Form.Item
                        name="MgO"
                        noStyle={true}
                      >
                        <InputNumber step={0.01} />
                      </Form.Item>
                      <Form.Item
                        name="priorityMgO"
                        noStyle={true}
                      >
                        <InputNumber placeholder="优先级" />
                      </Form.Item>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="模型参数" key="3" forceRender>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label="最大矿量"
                      name="modelFactorAlpha"
                      initialValue={1}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="最少剩余"
                      name="modelFactorBeta"
                      initialValue={1}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="元素含量"
                      name="modelFactorGamma"
                      initialValue={1}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber />
                    </Form.Item>
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label="优化算法种群数量"
                      name="gaPop"
                      initialValue={25}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="优化算法迭代次数"
                      name="gaEpoch"
                      initialValue={25}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber />
                    </Form.Item>
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
          </div>
          <div>
            <Button htmlType="submit" type="primary" style={{ width: '200px' }}>
              演&nbsp;&nbsp;算
            </Button>
          </div>
        </Form>
      </div>
      <Divider />
      <h1 style={{ textAlign: 'center', fontWeight: '900' }}>配料输出</h1>
      <Divider />
      <div style={{ marginTop: '20px' }}>
        {
          resultShow && (result === null ? <div style={{ paddingTop: 100, textAlign: 'center' }}>
            <Spin size="large" />
          </div> : <div>
              <div>
                <p>演算参数</p>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="氧料比(立方米/吨)"
                      value={result.calculateParameter.oxygenMaterialRatio}
                      disabled
                    />
                  </Col>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="总消耗(吨)"
                      value={result.calculateParameter.totalConsumedAmount}
                      disabled
                    />
                  </Col>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="总剩余(吨)"
                      value={result.calculateParameter.totalLeftOver}
                      disabled
                    />
                  </Col>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="目标得分"
                      value={result.calculateParameter.best_y}
                      disabled
                    />
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore={<span>一次风量m<sup>3</sup>/h</span>}
                      value={result.calculateParameter.paFlow}
                      disabled
                    />
                  </Col>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="S/Cu(%)"
                      value={result.calculateParameter.SCuRatio}
                      disabled
                    />
                  </Col>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="冰铜量(吨)"
                      value={result.calculateParameter.totalMatte}
                      disabled
                    />
                  </Col>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="渣量(吨)"
                      value={result.calculateParameter.totalSlag}
                      disabled
                    />
                  </Col>
                </Row>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Input
                      style={{ width: '250px' }}
                      addonBefore="石英石(吨)"
                      value={result.calculateParameter.totalQuartz}
                      disabled
                    />
                  </Col>
                </Row>
              </div>
              <div>
                <Button
                  type="primary"
                  style={{ width: '200px' }}
                  onClick={quickUpdate}
                >
                  更新
                  </Button>
              </div>
              <div>
                <div style={{ marginTop: '20px' }}>
                  <TableContext.Provider
                    value={{
                      columns: resultListColumns,
                      dataSource: result.list,
                      setData: setResultList
                    }}>
                    <EditTable />
                  </TableContext.Provider >
                </div>
              </div>
              <div style={{ marginTop: '20px' }}>
                <Table
                  rowKey={'name'}
                  columns={resultElementsMixtureListColumns}
                  dataSource={
                    (() => {
                      let value = { name: '参数' }
                      result.elementsMixtureList.forEach((item) => {
                        value[item.name] = item.percentage
                      })
                      console.log(value)
                      return [value]
                    })()
                  }
                  pagination={false}
                  bordered
                />
              </div>
            </div>)
        }
      </div>
    </div>
  );
}
function mapStateToProps(state) {
  const { config } = state.global;
  return {
    config
  };
}
export default connect(mapStateToProps)(H);
