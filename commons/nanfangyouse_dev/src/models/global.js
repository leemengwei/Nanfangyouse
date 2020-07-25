export default {
  namespace: 'global',
  state: {
    config: {
      host: '127.0.0.1',
      port: 7001
    },
    peifangList: [],
    pinghengParameter: {
      recoveryAu: 0,
      recoveryAg: 0,
      recoveryCu: 0,
      recoveryAu_: 0,
      recoveryAg_: 0,
      recoveryCu_: 0,
    },
    materialOptions: [],
    pinghengOldList: [],
    pinghengSetList: [],
    pinghengDataList: [],
    pinghengResizeDataList: []
  },
  reducers: {
    changeConfig(state, payload) {
      return {
        ...state,
        config: {
          host: payload.host || '127.0.0.1',
          port: payload.port || 7001
        }
      };
    },
    changePinghengOldList(state, payload) {
      return {
        ...state,
        pinghengOldList: [...payload.list]
      }
    },
    changeMaterialOptions(state, payload) {
      return {
        ...state,
        materialOptions: [...payload.list]
      }
    },
    changePinghengDataList(state, payload) {
      return {
        ...state,
        pinghengDataList: [...payload.list]
      }
    },
    changePinghengSetList(state, payload) {
      return {
        ...state,
        pinghengSetList: [...payload.settingList],
        pinghengDataList: [...payload.dataList]
      }
    },
    changePinghengParameter(state, payload) {
      const pinghengParameter = { ...state.pinghengParameter };
      Object.assign(pinghengParameter, payload.value);
      return {
        ...state,
        pinghengParameter
      }
    },
    changePinghengResizeDataList(state, payload) {
      return {
        ...state,
        pinghengResizeDataList: [...payload.list]
      }
    }
  },
}