import React, { Component } from 'react';
import './App.css';
import { Grid, Fab, Slide, Typography } from '@material-ui/core';
import AddPhotoAlternateIcon from "@material-ui/icons/AddPhotoAlternate";
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import axios from 'axios';
import {DropzoneArea} from 'material-ui-dropzone'

const font = "'Montserrat', sans-serif";

const muiTheme = createMuiTheme({
    typography : {
        fontFamily: font
    }
})

class App extends Component {

    componentWillMount() {
        this.setState({
            imageUploaded: 0
        })
    };

    handleUploadClick = event => {
        console.log(event.target.files.length);
        var files = [event.target.files[0], event.target.files[1]];//, event.target.files[2], event.target.files[3]];
        // var file = event.target.files[0];
        // const reader = new FileReader();
        // var url = reader.readAsDataURL(file);
    
        // reader.onloadend = function(e) {
        //   this.setState({
        //     selectedFile: [reader.result]
        //   });
        // }.bind(this);
    
        this.setState({
          selectedFile: files,
          imageUploaded: 1
        });

        var form = new FormData();
        files.forEach((file, index, arr)=> {
            form.append(index, file);
        });

        axios.post('/generateBinvox', form, {
            headers: {
              'accept': 'application/json',
              'Accept-Language': 'en-US,en;q=0.8',
              'Content-Type': `multipart/form-data; boundary=${form._boundary}`,
            }
        }).then((response) => {
            console.log(file)
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
            </div>
            </ThemeProvider>
        );
    };
}

export default App;
