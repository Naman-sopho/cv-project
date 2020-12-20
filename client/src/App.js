import React, { Component } from 'react';
import './App.css';
import { Grid, Fab, Slide, Typography } from '@material-ui/core';
import AddPhotoAlternateIcon from "@material-ui/icons/AddPhotoAlternate";
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import axios from 'axios';
import {DropzoneArea} from 'material-ui-dropzone'
import init_vox from './utils/render_binvox'

const font = "'Montserrat', sans-serif";

const muiTheme = createMuiTheme({
    typography : {
        fontFamily: font
    }
})

class App extends Component {

    componentWillMount() {
        this.setState({
            imageUploaded: 0,
            loading: false
        })
    };

    handleUploadClick = event => {
        if (event.target.files.length < 4) {
            return 0
        }

        var files = [event.target.files[0], event.target.files[1], event.target.files[2], event.target.files[3]];

        console.log(files)
        this.setState({
          selectedFile: files,
          imageUploaded: 1
        });

        var form = new FormData();
        files.forEach((file, index, arr)=> {
            form.append(index, file);
        });

        this.setState({
            loading: true
        })

        axios.post('/generateBinvox', form, {
            headers: {
              'accept': 'application/json',
              'Accept-Language': 'en-US,en;q=0.8',
              'Content-Type': `multipart/form-data; boundary=${form._boundary}`,
            }
        }).then((response_) => {
            axios.get('/generateBinvox', {responseType: 'blob'}).then((response) => {
                // const link = document.createElement('a');
                var binaryData = [];
                binaryData.push(response.data);
            var blob = new Blob([response.data], {type: 'application/octet-stream'});
            
            init_vox(blob);
            this.setState({
                loading: false
            })
        });
          })
          .catch((error) => {
            //handle error
          });
    };

    render() {
        return (
            <ThemeProvider theme={muiTheme}>
            <div className="App">
                <Grid className="App-header" container>
                    <Grid item md={12}>
                        <input
                            accept="image/*"
                            // className={classes.input}
                            id="contained-button-file"
                            multiple
                            hidden
                            type="file"
                            onChange={this.handleUploadClick}
                        />
                        <label htmlFor="contained-button-file">
                            <Fab component="span">
                                <AddPhotoAlternateIcon />
                            </Fab>
                        </label>
                    </Grid>
                    <Grid item md={12}>
                        {this.state.loading && <Typography>Generating model...will take a while!!!! :)</Typography>}
                    </Grid>
                    <Grid item md={8}>
                        <div id="3d-container" style={{"height": 500, "width": 500}}>
                        </div>
                        <div id="size">
                        </div>
                        <div id="error">
                        </div>
                    </Grid>
                </Grid>
            </div>
        </ThemeProvider>
        );
    };
}

export default App;
