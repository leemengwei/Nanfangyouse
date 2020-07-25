//import withRouter from 'umi/withRouter';

function BasicLayout(props) {
  console.log(props)
  return (
    <div>
      {props.children}
    </div>
  );
}

export default BasicLayout;
