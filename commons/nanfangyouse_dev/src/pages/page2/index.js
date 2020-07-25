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
  Switch,
  Table,
  Spin,
  Divider,
  Tabs,
  message
} from 'antd';
import { connect } from 'dva';
import request from '@/utils/request';
import { list2_1, columns1_2, columns1_3 } from '@/utils/data';
import styles from '../index.less';

const { TabPane } = Tabs;
const { TextArea } = Input;

const fkdata = [...list2_1];
const resultElementsMixtureListColumns = [...columns1_3];
const resultListColumns = columns1_2.filter((item) => {
  return item.dataIndex !== 'inventory'
});

function P(props) {
  const { config } = props;
  
  const [form] = Form.useForm();
  const [data, setData] = useState(fkdata);
  const [result, setResult] = useState(null);
  const [manual, setManual] = useState(false);
  const [materialList, setMaterialList] = useState([{ name: '配方1' }, { name: '配方2' }]);
  const [tableLoading, setTableLoading] = useState(false);
  const [resultShow, setResultShow] = useState(false);

  const setResultList = (data) => {
    setResult({ ...result, list: [...data] })
  }
  function setData_(payload, type) {
    const list = [...data];
    list[type] = payload;
    setData(list);
  }
  function setManual_(payload) {
    setManual(payload)
    // const newData = data.map(item => {
    //   return item.map(i => {
    //     const nI = i;
    //     'cohesion' in nI && (nI.cohesion = true);
    //     'delete' in nI && (nI.delete = true);
    //     return nI
    //   })
    // })
    // setData(newData)
  }
  const columns = [
    {
      title: '衔接',
      dataIndex: 'cohesion',
      render: (text, record, index) => <Checkbox
        checked={text}
        disabled={manual}
        onChange={() => {
          const newData = [...prevCountRef.current[0]];
          
          const canCheck = newData.some((item, i) => {
            if (i === index) {
              return false
            }
            return newData[index].name === item.name
          })
          if (canCheck) {
            return message.warning('对于新旧订单都有的物料，其衔接在实际料斗加料中自动完成')
          }
          newData[index] = { ...record }
          newData[index].cohesion = !text;
          setData_(newData, 0)
        }}
      />
    },
    {
      title: '排除',
      dataIndex: 'delete',
      render: (text, record, index) => <Checkbox
        checked={text}
        disabled={manual}
        onChange={() => {
          const newData = [...prevCountRef.current[1]];
          newData[index] = { ...record }
          newData[index].delete = !text
          console.log(newData)
          setData_(newData, 1)
        }}
      />
    },
    {
      title: '手选',
      dataIndex: 'manual',
      render: (text, record, index) => <Checkbox
        checked={text}
        disabled={!manual}
        onChange={() => {
          const newData = record.formula == 1 ? [...prevCountRef.current[0]] : [...prevCountRef.current[1]];
          newData[index] = { ...record }
          newData[index].manual = !text
          setData_(newData, record.formula == 1 ? 0: 1)
        }}
      />
    },
    {
      title: '所属配方',
      dataIndex: 'formula',
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
    },
    {
      title: 'Fe',
      dataIndex: 'Fe',
    },
    {
      title: 'S',
      dataIndex: 'S',
    },
    {
      title: 'SiO2',
      dataIndex: 'SiO2'
    }, {
      title: 'CaO',
      dataIndex: 'CaO',
    },
    {
      title: 'As',
      dataIndex: 'As',
    }, {
      title: 'Zn',
      dataIndex: 'Zn',
    },
    {
      title: 'Pb',
      dataIndex: 'Pb',
    },
    {
      title: 'MgO',
      dataIndex: 'MgO',
    },
    {
      title: 'Al2O3',
      dataIndex: 'Al2O3',
    },
    {
      title: 'H2O',
      dataIndex: 'H2O',
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
      title: '演算比例',
      dataIndex: 'calculatePercentage',
      editable: true,
      step: .01
    },
    {
      title: '',
      dataIndex: 'inventoryBalance',
      editable: true,
    },
    {
      title: '生产时间',
      dataIndex: 'productionTime',
    },
  ];
  let prevCountRef = useRef([...data]);
  useEffect(() => {
    const n = JSON.parse(JSON.stringify(data));
    prevCountRef.current = n;
  }, [data])

  function getInfo() {
    setTableLoading(true)
    request({
      method: 'GET',
      host: config.host,
      port: config.port,
      url: 'getFormula',
      cb: (res) => {
        const data = res.list;
        setData(data);
        const materialList = res.materialList.map((item) => {
          const o = {};
          item.elementsList.forEach((i) => {
            o[i.name] = i.percentage;
          })
          return {
            name: '配方' + item.formula,
            ...o
          }
        })
        form.setFieldsValue({
          formula1: res.oxygenMaterialRatio.formula1,
          formula2: res.oxygenMaterialRatio.formula2,
          'formula*': res.oxygenMaterialRatio['formula*']
        })
        setMaterialList(materialList);
        setTableLoading(false)
      }
    })
  }
  function onFinish(values) {
    const list = data.map(item => {
      return item.filter(i => {
        return !('delete' in i && i.delete === true)
      })
    })
    console.log(list)
    const payload = {
      list,
      oldList: JSON.parse(sessionStorage.getItem('page1list')),
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
      oxygenMaterialRatio: {
        formula1: values.formula1,
        formula2: values.formula2,
      }
    }
    console.log(payload)
    setResult(null)
    setResultShow(true)
    if (manual){
      const list = [...data.flat()].filter((item) => item.manual);
      const result_ = {
        list,
        calculateParameter: {
          oxygenMaterialRatio: '',
          totalConsumedAmount: '',
          paFlow: '',
          SCuRatio: '',
          totalMatte: '',
          totalSlag: '',
          totalQuartz: ''
        },
        elementsMixtureList: [],
        recommended: ''
      }
      return setResult(result_)
    }
    request({
      method: 'POST',
      host: config.host,
      port: config.port,
      url: 'quick_recommend',
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
      url: 'quick_update2',
      payload: result,
      cb: (res) => {
        setResult(res)
      }
    })
  }
  function onFinishFailed(err) {
    console.log(err)
  }

  return (
    <div>
      <div>
        <div style={{ marginBottom: '10px' }}>
          <Button type="primary"
            onClick={() => {
              getInfo()
            }}>
            获取订单
        </Button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' ,margin: '10px 0'}}>
          <span style={{fontSize: '14px', marginRight: '10px'}}>自动</span>
          <Switch checked={manual} onChange={setManual_} />
          <span style={{fontSize: '14px', marginLeft: '10px'}}>手动</span>
        </div>
        <Spin spinning={tableLoading}>
          <div>
          <TableContext.Provider value={{
            columns: (() => {
              const c = columns.slice(0, 1).concat(columns.slice(2));
              c[c.length - 2] = {
                title: '希望继续消耗的库存',
                dataIndex: c[c.length - 2].dataIndex
              };
              console.log(c)
              return c;
            })(),
            dataSource: data[0],
            setData: (payload) => {
              setData_(payload, 0)
            }
          }}>
            <EditTable />
          </TableContext.Provider >
          </div>
          <div style={{marginTop: '20px'}}>
          <TableContext.Provider value={{
            columns: (() => {
              const c = columns.slice(1, 2).concat(columns.slice(2));
              c[c.length - 2] = {
                title: '配方2生产后理论剩余',
                dataIndex: c[c.length - 2].dataIndex
              };
              return c
            })(),
            dataSource: data[1],
            setData: (payload) => {
              setData_(payload, 1)
            }
          }}>
            <EditTable />
          </TableContext.Provider >
          </div>
          <Table
            className={styles.block}
            rowKey={'name'}
            columns={columns1_3}
            dataSource={materialList}
            pagination={false}
            bordered
          />
        </Spin>
      </div>
      <div>
        <Form
          form={form}
          className={styles.block}
          labelCol={{ span: 12 }}
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
                      <InputNumber step={0.01} disabled/>
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="冰铜铁含量(%)"
                      name="matteFePercentage"
                      initialValue={3.5}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="渣中二氧化硅含量(%)"
                      name="slagSiO2Percentage"
                      initialValue={24}
                      rules={[
                        {
                          required: true
                        },
                      ]}
                    >
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber step={0.01} disabled/>
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
                      <InputNumber disabled/>
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
                      <InputNumber step={0.01} disabled/>
                    </Form.Item>
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
            <Tabs defaultActiveKey="1" type="card">
              <TabPane tab="氧料比" key="1" forceRender>
                <Row className={styles.row}>
                  <Col span={6}>
                    <Form.Item
                      label="配方1"
                      name="formula1"
                    >
                      <Input disabled />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="配方2"
                      name="formula2"
                    >
                      <Input disabled />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item
                      label="配方*"
                      name="formula*"
                    >
                      <Input disabled/>
                    </Form.Item>
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
          </div>
          <div>
            <Button htmlType="submit" type="primary" style={{ width: '200px' }}>
              {manual ? '确定手选内容' : '推荐'}
            </Button>
          </div>
        </Form>
        <div>
          <Divider />
          <h1 style={{ textAlign: 'center', fontWeight: '900' }}>衔接输出</h1>
          <Divider />
          <div style={{ marginTop: '20px' }}>
            {
              resultShow && (result === null ? <div style={{ paddingTop: 100, textAlign: 'center' }}>
                <Spin size="large" />
              </div> : <div>
                  <div>
                    <p style={{fontSize: '18px'}}>结果说明</p>
                    <TextArea
                      autoSize
                      value={result.recommended}
                    />
                  </div>
                  <div style={{ marginTop: '20px' }}>
                    <p>演算参数</p>
                    <Row className={styles.row}>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore={<span>氧料比(Nm<sup>3</sup>/t)</span>}
                          value={result.calculateParameter?.oxygenMaterialRatio}
                          disabled
                        />
                      </Col>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore="总消耗(吨)"
                          value={result.calculateParameter?.totalConsumedAmount}
                          disabled
                        />
                      </Col>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore="总剩余(吨)"
                          value={result.calculateParameter?.totalLeftOver}
                          disabled
                        />
                      </Col>
                    </Row>
                    <Row className={styles.row}>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore={<span>一次风量m<sup>3</sup>/h</span>}
                          value={result.calculateParameter?.paFlow}
                          disabled
                        />
                      </Col>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore="S/Cu(%)"
                          value={result.calculateParameter?.SCuRatio}
                          disabled
                        />
                      </Col>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore="冰铜量(吨)"
                          value={result.calculateParameter?.totalMatte}
                          disabled
                        />
                      </Col>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore="渣量(吨)"
                          value={result.calculateParameter?.totalSlag}
                          disabled
                        />
                      </Col>
                    </Row>
                    <Row className={styles.row}>
                      <Col span={6}>
                        <Input
                          style={{ width: '250px' }}
                          addonBefore="石英石(吨)"
                          value={result.calculateParameter?.totalQuartz}
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
                      dataSource={(() => {
                        let value = { name: '参数' }
                        result.elementsMixtureList.forEach((item) => {
                          value[item.name] = item.percentage
                        })
                        return [value]
                      })()}
                      pagination={false}
                      bordered
                    />
                  </div>
                </div>)
            }
          </div>
        </div>
      </div>
    </div>
  )
}

function mapStateToProps(state) {
  const { config } = state.global;
  return {
    config
  };
}
export default connect(mapStateToProps)(P);